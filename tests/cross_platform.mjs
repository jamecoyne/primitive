// Cross-platform pixel-diff test.
//
// Renders the same locked frame on:
//   - native, via `cargo run --bin render_frame` (wgpu → naga → MSL → Metal)
//   - browser, via Playwright + WebGPU (wgpu → naga → WGSL → browser → MSL)
// then strict-compares pixels. Fails if any channel of any pixel differs.
//
// Why we expect this to fail today: the two paths produce subtly different
// floats — both go through naga as the IR but the *output* shader language
// differs (MSL native vs WGSL handed to the browser), and the browser's
// canvas → screenshot pipeline applies its own composite. The test exists
// to (a) quantify the divergence so we know if it grows, and (b) be the
// first place to find out if a future change makes the platforms agree
// pixel-perfect.

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

const W = 960, H = 720;
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

function renderNative(outputPath) {
    const r = spawnSync('cargo', [
        'run', '--bin', 'render_frame', '--quiet', '--',
        '--output', outputPath,
        '--time', String(TIME),
        '--mx', String(MX),
        '--my', String(MY),
        '--width', String(W),
        '--height', String(H),
    ], { stdio: ['ignore', 'inherit', 'inherit'], cwd: ROOT });
    if (r.status !== 0) {
        console.error('cargo run --bin render_frame failed');
        process.exit(2);
    }
}

async function renderWeb(outputPath) {
    const { server, url } = await startServer();
    const browser = await chromium.launch({
        args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan',
            '--use-angle=metal',
            '--ignore-gpu-blocklist',
            '--enable-gpu-rasterization',
            // Force sRGB color profile for screenshots so the captured bytes
            // match what the GPU wrote — without this Chromium may convert
            // through the display profile, shifting magenta channel values
            // by ~32/255 vs the native readback.
            '--force-color-profile=srgb',
            '--disable-features=ColorCorrectRendering',
        ],
    });
    const context = await browser.newContext({
        viewport: { width: 1024, height: 800 },
        deviceScaleFactor: 1,
    });
    // Strip browser-injected canvas chrome before paint:
    //  - box-shadow: orange outer shadow that anti-aliases into edge pixels.
    //  - outline:    Chromium paints a 1px focus outline on the canvas
    //                (rgb(0,95,204), macOS system blue) which writes into
    //                the captured perimeter and would account for ~3.4k of
    //                edge-only diffs otherwise.
    await context.addInitScript(() => {
        const apply = () => {
            const style = document.createElement('style');
            style.textContent =
                'canvas#wgpu { box-shadow: none !important; outline: none !important; border: none !important; }';
            document.head.appendChild(style);
        };
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', apply, { once: true });
        } else {
            apply();
        }
    });
    const page = await context.newPage();

    const errors = [];
    page.on('pageerror', e => errors.push(`[pageerror] ${e.message}`));
    page.on('console', m => { if (m.type() === 'error') errors.push(`[error] ${m.text()}`); });

    await page.goto(`${url}?t=${TIME}&mx=${MX}&my=${MY}`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(2500);

    // Element-scoped screenshot: Playwright captures only the canvas element's
    // own layer, avoiding the page-compositor edge anti-aliasing that a
    // page-level screenshot+clip would mix in.
    await page.locator('canvas#wgpu').screenshot({ path: outputPath });

    await browser.close();
    server.close();

    if (errors.length > 0) {
        console.error('errors during web render:');
        errors.forEach(e => console.error('  ' + e));
        process.exit(2);
    }
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

    console.log('rendering native via cargo run --bin render_frame...');
    renderNative(nativePath);

    console.log('rendering web via headless Chromium...');
    await renderWeb(webPath);

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
