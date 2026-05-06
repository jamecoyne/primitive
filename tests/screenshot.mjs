// Headless screenshot test for the wasm build.
//
// Spins up a local static server for web/dist/, launches Chromium, waits for
// the wasm to render, screenshots the page, and analyzes pixel brightness.
// Saves diagnostics to tests/output/ for inspection.
//
// Exit code:
//   0 — canvas is rendering (avg brightness > MIN_AVG and color variance > MIN_VAR)
//   1 — canvas is black/uniform (the bug we're chasing)
//   2 — infra error (server, browser, build missing)

import { chromium } from 'playwright';
import { PNG } from 'pngjs';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, stat } from 'node:fs/promises';
import { extname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(import.meta.url), '../..');
const DIST = join(ROOT, 'web/dist');
const OUT  = join(ROOT, 'tests/output');

const MIME = {
    '.html': 'text/html; charset=utf-8',
    '.js':   'application/javascript; charset=utf-8',
    '.wasm': 'application/wasm',
    '.css':  'text/css; charset=utf-8',
    '.png':  'image/png',
    '.json': 'application/json; charset=utf-8',
};

async function ensureBuilt() {
    try {
        await stat(join(DIST, 'wgpuweb.js'));
        await stat(join(DIST, 'wgpuweb_bg.wasm'));
        await stat(join(DIST, 'index.html'));
    } catch {
        console.error('web/dist not built — run ./build-web.sh first');
        process.exit(2);
    }
}

function startServer() {
    const server = createServer(async (req, res) => {
        let urlPath = decodeURIComponent(new URL(req.url, 'http://x').pathname);
        if (urlPath === '/') urlPath = '/index.html';
        const filePath = join(DIST, urlPath);
        if (!filePath.startsWith(DIST)) { res.writeHead(403).end(); return; }
        try {
            const buf = await readFile(filePath);
            const mime = MIME[extname(filePath)] ?? 'application/octet-stream';
            res.writeHead(200, {
                'Content-Type': mime,
                // COOP/COEP for cross-origin isolation (needed by some wasm setups).
                'Cross-Origin-Opener-Policy': 'same-origin',
                'Cross-Origin-Embedder-Policy': 'require-corp',
            });
            res.end(buf);
        } catch {
            res.writeHead(404).end('not found');
        }
    });
    return new Promise(resolve_ => {
        server.listen(0, '127.0.0.1', () => {
            const port = server.address().port;
            resolve_({ server, url: `http://127.0.0.1:${port}/` });
        });
    });
}

function analyzePng(buf) {
    const png = PNG.sync.read(buf);
    let totalLuma = 0;
    let minR = 255, maxR = 0, minG = 255, maxG = 0, minB = 255, maxB = 0;
    const pixels = png.width * png.height;
    for (let i = 0; i < png.data.length; i += 4) {
        const r = png.data[i], g = png.data[i+1], b = png.data[i+2];
        totalLuma += 0.299 * r + 0.587 * g + 0.114 * b;
        if (r < minR) minR = r; if (r > maxR) maxR = r;
        if (g < minG) minG = g; if (g > maxG) maxG = g;
        if (b < minB) minB = b; if (b > maxB) maxB = b;
    }
    return {
        width: png.width,
        height: png.height,
        avgLuma: totalLuma / pixels,
        rangeR: maxR - minR,
        rangeG: maxG - minG,
        rangeB: maxB - minB,
        png,
    };
}

// Mean absolute difference across RGB channels of two same-size PNG buffers.
function meanAbsDiff(pngA, pngB) {
    if (pngA.width !== pngB.width || pngA.height !== pngB.height) {
        throw new Error(`size mismatch ${pngA.width}x${pngA.height} vs ${pngB.width}x${pngB.height}`);
    }
    let sum = 0;
    let n = 0;
    for (let i = 0; i < pngA.data.length; i += 4) {
        sum += Math.abs(pngA.data[i]   - pngB.data[i])
            +  Math.abs(pngA.data[i+1] - pngB.data[i+1])
            +  Math.abs(pngA.data[i+2] - pngB.data[i+2]);
        n += 3;
    }
    return sum / n;
}

async function main() {
    await ensureBuilt();
    await mkdir(OUT, { recursive: true });

    const { server, url } = await startServer();
    console.log(`serving ${DIST} at ${url}`);

    const browser = await chromium.launch({
        // Try WebGPU first; fall back to WebGL2 if not available.
        args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--use-angle=metal',
            '--ignore-gpu-blocklist',
            '--enable-gpu-rasterization',
        ],
    });

    const context = await browser.newContext({
        viewport: { width: 1024, height: 800 },
        deviceScaleFactor: 1,
    });
    const page = await context.newPage();

    const consoleLog = [];
    page.on('console', msg => consoleLog.push(`[${msg.type()}] ${msg.text()}`));
    page.on('pageerror', err => consoleLog.push(`[pageerror] ${err.message}\n${err.stack ?? ''}`));
    page.on('requestfailed', req =>
        consoleLog.push(`[netfail] ${req.url()} -> ${req.failure()?.errorText}`));

    await page.goto(url, { waitUntil: 'networkidle' });
    // Give wgpu a moment to spin up and render a few frames.
    await page.waitForTimeout(2500);

    const canvasInfo = await page.evaluate(() => {
        const c = document.querySelector('canvas');
        if (!c) return { exists: false };
        const r = c.getBoundingClientRect();
        return {
            exists: true,
            width: c.width,
            height: c.height,
            clientWidth: c.clientWidth,
            clientHeight: c.clientHeight,
            inDom: document.contains(c),
            display: getComputedStyle(c).display,
            rect: { x: r.x, y: r.y, w: r.width, h: r.height },
        };
    });

    const gpuStatus = await page.evaluate(async () => {
        const out = { hasWebGPU: !!navigator.gpu, adapter: null };
        if (navigator.gpu) {
            try {
                const a = await navigator.gpu.requestAdapter();
                out.adapter = a ? (a.info ? { vendor: a.info.vendor, arch: a.info.architecture } : 'present') : 'null';
            } catch (e) { out.adapter = 'err: ' + e.message; }
        }
        return out;
    });

    const fullScreenshotPath = join(OUT, 'screenshot.png');
    await page.screenshot({ path: fullScreenshotPath, fullPage: false });

    if (!canvasInfo.exists || canvasInfo.rect.w <= 0 || canvasInfo.rect.h <= 0) {
        await writeFile(join(OUT, 'console.log'), consoleLog.join('\n'));
        console.error('canvas not found or zero-sized; aborting');
        await browser.close();
        server.close();
        process.exit(1);
    }

    const clip = {
        x: Math.round(canvasInfo.rect.x),
        y: Math.round(canvasInfo.rect.y),
        width: Math.round(canvasInfo.rect.w),
        height: Math.round(canvasInfo.rect.h),
    };

    // Baseline: cursor un-moved (state.mouse defaults to canvas center).
    const baselinePath = join(OUT, 'canvas.png');
    const baselineBuf = await page.screenshot({ path: baselinePath, clip });
    const stats = analyzePng(baselineBuf);

    // Mouse-API check: move cursor near top-left of the canvas. The
    // mandelbrot's center should track the cursor, so the rendered image
    // should differ noticeably from the baseline.
    await page.mouse.move(canvasInfo.rect.x + 80, canvasInfo.rect.y + 80);
    await page.waitForTimeout(400);
    const mousedPath = join(OUT, 'canvas-mouse.png');
    const mousedBuf = await page.screenshot({ path: mousedPath, clip });
    const mousedStats = analyzePng(mousedBuf);
    const mouseDiff = meanAbsDiff(stats.png, mousedStats.png);

    await writeFile(join(OUT, 'console.log'), consoleLog.join('\n'));

    const { png: _p1, ...statsLog }       = stats;
    const { png: _p2, ...mousedStatsLog } = mousedStats;

    console.log('\n=== diagnostics ===');
    console.log('canvas:        ', canvasInfo);
    console.log('gpu:           ', gpuStatus);
    console.log('baseline image:', statsLog);
    console.log('after mouse:   ', mousedStatsLog);
    console.log(`mouse-move mean abs diff: ${mouseDiff.toFixed(2)}/255`);
    console.log(`baseline screenshot:  ${baselinePath}`);
    console.log(`after-mouse screenshot: ${mousedPath}`);
    console.log(`console (${consoleLog.length} lines): tests/output/console.log`);
    if (consoleLog.length) {
        console.log('--- console (first 30 lines) ---');
        consoleLog.slice(0, 30).forEach(l => console.log('  ' + l));
    }

    await browser.close();
    server.close();

    const MIN_AVG = 8;          // any visible content beats pure black
    const MIN_RANGE = 30;       // non-uniform color implies mandelbrot is rendering
    // Time-based color/zoom drift between two ~400ms screenshots is well
    // under 10/255 mean abs diff; threshold of 12 leaves margin while still
    // catching a mouse-uniform that doesn't actually translate the image.
    const MIN_MOUSE_DIFF = 12;

    const renderingOk = stats.avgLuma >= MIN_AVG &&
                        (stats.rangeR + stats.rangeG + stats.rangeB) >= MIN_RANGE;
    const mouseOk = mouseDiff >= MIN_MOUSE_DIFF;

    if (!renderingOk) {
        console.error(`\n✗ canvas appears black/uniform (avgLuma=${stats.avgLuma.toFixed(2)}, ranges=${stats.rangeR}/${stats.rangeG}/${stats.rangeB})`);
        process.exit(1);
    }
    if (!mouseOk) {
        console.error(`\n✗ mouse API not affecting render (mean abs diff ${mouseDiff.toFixed(2)} < ${MIN_MOUSE_DIFF})`);
        process.exit(1);
    }
    console.log(`\n✓ rendering ok (avgLuma=${stats.avgLuma.toFixed(2)}), mouse responsive (diff=${mouseDiff.toFixed(2)})`);
}

main().catch(e => {
    console.error('test infra error:', e);
    process.exit(2);
});
