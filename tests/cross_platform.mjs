// Cross-platform pixel-diff test.
//
// Renders the same locked frame on:
//   - native, via `cargo run --bin render_frame` (wgpu → naga → MSL → Metal)
//   - browser, via Playwright + WebGPU (wgpu → naga → WGSL → browser → MSL)
// then strict-compares pixels. Fails if any channel of any pixel differs.
//
// The web canvas now follows the viewport size, so we render web first to
// learn its actual dimensions, then run the native renderer at the same
// dimensions before comparing.

import { chromium } from 'playwright';
import { PNG } from 'pngjs';
import { spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { Buffer } from 'node:buffer';
import { readFile, mkdir, stat, writeFile } from 'node:fs/promises';
import { extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(import.meta.url), '../..');
const DIST = join(ROOT, 'web/dist');
const OUT  = join(ROOT, 'tests/output');

// Viewport size — canvas fills 100vw × 100vh in CSS, so canvas dimensions
// (and therefore the native bin's render size) follow this.
const VIEWPORT_W = 1024;
const VIEWPORT_H = 800;
const TIME = 2.5;
const MX = 0.5;
const MY = 0.5;

const MIME = {
    '.html': 'text/html; charset=utf-8',
    '.js':   'application/javascript; charset=utf-8',
    '.wasm': 'application/wasm',
    '.css':  'text/css; charset=utf-8',
    '.png':  'image/png',
};

function startServer() {
    const server = createServer(async (req, res) => {
        let urlPath = decodeURIComponent(new URL(req.url, 'http://x').pathname);
        if (urlPath === '/') urlPath = '/index.html';
        const filePath = join(DIST, urlPath);
        if (!filePath.startsWith(DIST)) { res.writeHead(403).end(); return; }
        try {
            const buf = await readFile(filePath);
            res.writeHead(200, {
                'Content-Type': MIME[extname(filePath)] ?? 'application/octet-stream',
                'Cross-Origin-Opener-Policy': 'same-origin',
                'Cross-Origin-Embedder-Policy': 'require-corp',
            });
            res.end(buf);
        } catch {
            res.writeHead(404).end();
        }
    });
    return new Promise(r => {
        server.listen(0, '127.0.0.1', () => r({ server, url: `http://127.0.0.1:${server.address().port}/` }));
    });
}

async function ensureBuilt() {
    try {
        await stat(join(DIST, 'wgpuweb.js'));
    } catch {
        console.error('web/dist not built — run ./build-web.sh first');
        process.exit(2);
    }
}

function renderNative(outputPath, width, height) {
    const r = spawnSync('cargo', [
        'run', '--bin', 'render_frame', '--quiet', '--',
        '--output', outputPath,
        '--time', String(TIME),
        '--mx', String(MX),
        '--my', String(MY),
        '--width', String(width),
        '--height', String(height),
    ], { stdio: ['ignore', 'inherit', 'inherit'], cwd: ROOT });
    if (r.status !== 0) {
        console.error('cargo run --bin render_frame failed');
        process.exit(2);
    }
}

// Returns { width, height } of the canvas as it actually rendered.
async function renderWeb(outputPath) {
    const { server, url } = await startServer();
    const browser = await chromium.launch({
        args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--use-angle=metal',
            '--ignore-gpu-blocklist',
            '--enable-gpu-rasterization',
            '--force-color-profile=srgb',
            '--disable-features=ColorCorrectRendering',
        ],
    });
    const context = await browser.newContext({
        viewport: { width: VIEWPORT_W, height: VIEWPORT_H },
        deviceScaleFactor: 1,
    });
    const page = await context.newPage();

    const errors = [];
    page.on('pageerror', e => errors.push(`[pageerror] ${e.message}`));
    page.on('console', m => { if (m.type() === 'error') errors.push(`[error] ${m.text()}`); });

    await page.goto(`${url}?t=${TIME}&mx=${MX}&my=${MY}`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(2500);

    const dims = await page.evaluate(() => {
        const c = document.querySelector('canvas');
        return c ? { width: c.width, height: c.height } : null;
    });
    if (!dims || dims.width <= 0 || dims.height <= 0) {
        console.error(`unexpected canvas size: ${JSON.stringify(dims)}`);
        process.exit(2);
    }

    // Element-scoped screenshot — captures only the canvas's own layer,
    // bypassing page-compositor edge anti-aliasing.
    await page.locator('canvas#wgpu').screenshot({ path: outputPath });

    await browser.close();
    server.close();

    if (errors.length > 0) {
        console.error('errors during web render:');
        errors.forEach(e => console.error('  ' + e));
        process.exit(2);
    }
    return dims;
}

function diff(nativePng, webPng) {
    if (nativePng.width !== webPng.width || nativePng.height !== webPng.height) {
        throw new Error(`size mismatch ${nativePng.width}x${nativePng.height} vs ${webPng.width}x${webPng.height}`);
    }
    const total = nativePng.width * nativePng.height;
    let differingPixels = 0;
    let maxChanDiff = 0;
    let totalChanDiff = 0;
    // Per-channel histograms of |diff|, capped at 32 buckets.
    const hist = new Uint32Array(33);
    for (let i = 0; i < nativePng.data.length; i += 4) {
        const dR = Math.abs(nativePng.data[i]   - webPng.data[i]);
        const dG = Math.abs(nativePng.data[i+1] - webPng.data[i+1]);
        const dB = Math.abs(nativePng.data[i+2] - webPng.data[i+2]);
        if (dR | dG | dB) differingPixels++;
        const m = Math.max(dR, dG, dB);
        if (m > maxChanDiff) maxChanDiff = m;
        totalChanDiff += dR + dG + dB;
        hist[Math.min(m, 32)]++;
    }
    return {
        totalPixels: total,
        differingPixels,
        maxChannelDiff: maxChanDiff,
        meanChannelDiff: totalChanDiff / (total * 3),
        hist: Array.from(hist),
    };
}

async function main() {
    await ensureBuilt();
    await mkdir(OUT, { recursive: true });

    const nativePath = join(OUT, 'native-frame.png');
    const webPath    = join(OUT, 'web-frame.png');

    console.log('rendering web via headless Chromium...');
    const { width, height } = await renderWeb(webPath);
    console.log(`web canvas: ${width}×${height}`);

    console.log(`rendering native via cargo run --bin render_frame at ${width}×${height}...`);
    renderNative(nativePath, width, height);

    const nativeBuf = await readFile(nativePath);
    const webBuf    = await readFile(webPath);
    const native = PNG.sync.read(nativeBuf);
    const web    = PNG.sync.read(webBuf);

    const stats = diff(native, web);
    const pct = (stats.differingPixels / stats.totalPixels * 100).toFixed(2);

    console.log('\n=== cross-platform diff ===');
    console.log(`differing pixels:    ${stats.differingPixels.toLocaleString()} / ${stats.totalPixels.toLocaleString()} (${pct}%)`);
    console.log(`max channel diff:    ${stats.maxChannelDiff} / 255`);
    console.log(`mean channel diff:   ${stats.meanChannelDiff.toFixed(3)} / 255`);
    console.log(`outputs:             tests/output/{native-frame,web-frame}.png`);

    // Histogram of per-pixel max channel diff.
    console.log('\nper-pixel max channel diff histogram (capped at 32):');
    for (let i = 0; i < stats.hist.length; i++) {
        if (stats.hist[i] > 0) {
            const bar = '█'.repeat(Math.min(40, Math.round(stats.hist[i] / stats.totalPixels * 100)));
            console.log(`  diff=${i.toString().padStart(2)}: ${stats.hist[i].toLocaleString().padStart(8)} ${bar}`);
        }
    }

    if (stats.differingPixels > 0) {
        console.error(
            `\n✗ ${stats.differingPixels} pixel(s) differ between native and web ` +
            `(strict cross-platform check)`,
        );
        process.exit(1);
    }
    console.log('\n✓ pixel-perfect cross-platform match');
}

main().catch(e => {
    console.error('test infra error:', e);
    process.exit(2);
});
