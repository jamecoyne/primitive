// Headless screenshot harness for the wasm build.
//
// Three checks:
//   1. Console-error gate     — any [error] / [pageerror] line fails.
//   2. Pinned-baseline diff   — fixed time + mouse uv → deterministic frame,
//                                compared to tests/baseline.png. Catches
//                                shader regressions, surface format drift.
//   3. Mouse-API responsive   — image must change after page.mouse.move(...).
//
// Build the wasm first (./build-web.sh). Run with `npm test`. Run with
// `UPDATE_BASELINE=1 npm test` to regenerate tests/baseline.png from the
// current renderer.

import { chromium } from 'playwright';
import { PNG } from 'pngjs';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, stat } from 'node:fs/promises';
import { extname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(import.meta.url), '../..');
const DIST = join(ROOT, 'web/dist');
const OUT  = join(ROOT, 'tests/output');
const BASELINE_PATH = join(ROOT, 'tests/baseline.png');

// URL params that lock time + mouse for a deterministic frame.
const BASELINE_PARAMS = 't=2.5&mx=0.5&my=0.5';
// For the mouse-API check we lock time but leave the mouse free so the
// diff between the two screenshots reflects only the cursor move, not
// time-based zoom/color drift.
const MOUSE_TEST_PARAMS = 't=2.5';

const MIN_AVG_LUMA   = 8;   // catches all-black canvas
const MIN_RGB_RANGE  = 30;  // catches uniform-colour canvas
const MIN_MOUSE_DIFF = 12;  // mouse move must shift image
const MAX_BASELINE_DIFF = 8; // tuned for same-host runs; bump if cross-host

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

// Loads the page at `${url}?${params}`, waits for first render, returns
// canvas info + clip rect for screenshots.
async function navigateAndWait(page, url, params) {
    const target = params ? `${url}?${params}` : url;
    await page.goto(target, { waitUntil: 'networkidle' });
    await page.waitForTimeout(2500);

    const info = await page.evaluate(() => {
        const c = document.querySelector('canvas');
        if (!c) return { exists: false };
        const r = c.getBoundingClientRect();
        return {
            exists: true,
            width: c.width, height: c.height,
            rect: { x: r.x, y: r.y, w: r.width, h: r.height },
        };
    });
    if (!info.exists || info.rect.w <= 0 || info.rect.h <= 0) {
        throw new Error('canvas missing or zero-sized');
    }
    info.clip = {
        x: Math.round(info.rect.x),
        y: Math.round(info.rect.y),
        width: Math.round(info.rect.w),
        height: Math.round(info.rect.h),
    };
    return info;
}

async function main() {
    await ensureBuilt();
    await mkdir(OUT, { recursive: true });

    const updateBaseline = process.env.UPDATE_BASELINE === '1';

    const { server, url } = await startServer();
    console.log(`serving ${DIST} at ${url}`);

    const browser = await chromium.launch({
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

    const failures = [];

    // ------------------------------------------------------------------
    // Phase A — pinned baseline (locked time + mouse)
    // ------------------------------------------------------------------
    const a = await navigateAndWait(page, url, BASELINE_PARAMS);
    const lockedPath = join(OUT, 'locked.png');
    const lockedBuf = await page.screenshot({ path: lockedPath, clip: a.clip });
    const lockedStats = analyzePng(lockedBuf);

    let baselineDiff = null;
    if (updateBaseline) {
        await writeFile(BASELINE_PATH, lockedBuf);
        console.log(`wrote baseline → ${BASELINE_PATH}`);
    } else {
        let baselineBuf;
        try {
            baselineBuf = await readFile(BASELINE_PATH);
        } catch {
            failures.push(
                `tests/baseline.png missing — run \`UPDATE_BASELINE=1 npm test\` to create it`,
            );
        }
        if (baselineBuf) {
            const baselinePng = PNG.sync.read(baselineBuf);
            try {
                baselineDiff = meanAbsDiff(baselinePng, lockedStats.png);
            } catch (e) {
                failures.push(`baseline size mismatch: ${e.message}`);
            }
        }
    }

    // Sanity: locked frame should still pass the brightness/range checks.
    if (lockedStats.avgLuma < MIN_AVG_LUMA ||
        (lockedStats.rangeR + lockedStats.rangeG + lockedStats.rangeB) < MIN_RGB_RANGE) {
        failures.push(
            `locked frame appears black/uniform ` +
            `(avgLuma=${lockedStats.avgLuma.toFixed(2)}, ranges=${lockedStats.rangeR}/${lockedStats.rangeG}/${lockedStats.rangeB})`,
        );
    }

    if (baselineDiff !== null && baselineDiff > MAX_BASELINE_DIFF) {
        failures.push(
            `pinned-baseline diff ${baselineDiff.toFixed(2)} > ${MAX_BASELINE_DIFF}/255 ` +
            `(see tests/output/locked.png vs tests/baseline.png)`,
        );
    }

    // ------------------------------------------------------------------
    // Phase B — mouse API responsiveness (locked time only)
    // ------------------------------------------------------------------
    const b = await navigateAndWait(page, url, MOUSE_TEST_PARAMS);
    const beforePath = join(OUT, 'before-mouse.png');
    const beforeBuf = await page.screenshot({ path: beforePath, clip: b.clip });
    const beforeStats = analyzePng(beforeBuf);

    await page.mouse.move(b.rect.x + 80, b.rect.y + 80);
    await page.waitForTimeout(400);
    const afterPath = join(OUT, 'after-mouse.png');
    const afterBuf = await page.screenshot({ path: afterPath, clip: b.clip });
    const afterStats = analyzePng(afterBuf);
    const mouseDiff = meanAbsDiff(beforeStats.png, afterStats.png);

    if (mouseDiff < MIN_MOUSE_DIFF) {
        failures.push(
            `mouse API not affecting render: diff ${mouseDiff.toFixed(2)} < ${MIN_MOUSE_DIFF}/255 ` +
            `(see tests/output/before-mouse.png vs after-mouse.png)`,
        );
    }

    // ------------------------------------------------------------------
    // Phase C — console-error gate (errors anywhere across the run)
    // ------------------------------------------------------------------
    await writeFile(join(OUT, 'console.log'), consoleLog.join('\n'));
    const errorLines = consoleLog.filter(l =>
        l.startsWith('[error]') || l.startsWith('[pageerror]') || l.startsWith('[netfail]')
    );
    if (errorLines.length > 0) {
        failures.push(`${errorLines.length} console error(s):\n  ` + errorLines.slice(0, 5).join('\n  '));
    }

    // ------------------------------------------------------------------
    // Diagnostics
    // ------------------------------------------------------------------
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

    const stripPng = ({ png, ...rest }) => rest;
    console.log('\n=== diagnostics ===');
    console.log('gpu:           ', gpuStatus);
    console.log('locked frame:  ', stripPng(lockedStats));
    console.log('after-mouse:   ', stripPng(afterStats));
    console.log(`baseline diff:  ${baselineDiff === null ? '(skipped)' : baselineDiff.toFixed(2) + '/255'}`);
    console.log(`mouse diff:     ${mouseDiff.toFixed(2)}/255`);
    console.log(`outputs:        tests/output/{locked,before-mouse,after-mouse}.png`);
    console.log(`console:        ${consoleLog.length} lines → tests/output/console.log`);

    await browser.close();
    server.close();

    if (failures.length > 0) {
        console.error('\n✗ FAIL:');
        failures.forEach(f => console.error('  - ' + f));
        process.exit(1);
    }

    if (updateBaseline) {
        console.log('\n✓ baseline updated; commit tests/baseline.png');
    } else {
        console.log('\n✓ all checks passed');
    }
}

main().catch(e => {
    console.error('test infra error:', e);
    process.exit(2);
});
