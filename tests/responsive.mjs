// Responsive behaviour — checks the wasm canvas (a) fills the viewport at
// physical-pixel resolution under high-DPR, and (b) tracks viewport resizes
// via winit's ResizeObserver. Both bugs were observed live: the page only
// rendered into a quarter of the screen on a Retina display, and the canvas
// did not resize when the browser window resized.
//
// Two phases:
//   1. DPR=2: drawing buffer must be viewport × 2; CSS size = viewport.
//      Element-cropped screenshot must be effectively full (no big black
//      stripes from a too-small canvas).
//   2. Resize: launch at one viewport, capture canvas dims, setViewportSize
//      to a different viewport, capture again, assert dims updated to match
//      the new viewport (× DPR).

import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { PNG } from 'pngjs';
import { readFile, mkdir, stat, writeFile } from 'node:fs/promises';
import { extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(import.meta.url), '../..');
const DIST = join(ROOT, 'web/dist');
const OUT  = join(ROOT, 'tests/output');

const MIME = {
    '.html': 'text/html; charset=utf-8',
    '.js':   'application/javascript; charset=utf-8',
    '.wasm': 'application/wasm',
    '.css':  'text/css; charset=utf-8',
};

function startServer() {
    const server = createServer(async (req, res) => {
        let p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
        if (p === '/') p = '/index.html';
        const filePath = join(DIST, p);
        if (!filePath.startsWith(DIST)) { res.writeHead(403).end(); return; }
        try {
            const buf = await readFile(filePath);
            res.writeHead(200, {
                'Content-Type': MIME[extname(filePath)] ?? 'application/octet-stream',
            });
            res.end(buf);
        } catch {
            res.writeHead(404).end();
        }
    });
    return new Promise(r => {
        server.listen(0, '127.0.0.1', () => r({
            server, url: `http://127.0.0.1:${server.address().port}/`,
        }));
    });
}

async function ensureBuilt() {
    try { await stat(join(DIST, 'wgpuweb.js')); }
    catch {
        console.error('web/dist not built — run ./build-web.sh first');
        process.exit(2);
    }
}

async function readCanvasState(page) {
    return await page.evaluate(() => {
        const c = document.querySelector('canvas#wgpu');
        if (!c) return null;
        const r = c.getBoundingClientRect();
        return {
            buffer: { w: c.width, h: c.height },
            css:    { w: c.clientWidth, h: c.clientHeight },
            rect:   { w: r.width, h: r.height },
            dpr:    window.devicePixelRatio,
            view:   { w: window.innerWidth, h: window.innerHeight },
        };
    });
}

// Brightness sanity — element-cropped canvas screenshot should be visibly
// rendered (mandelbrot has ~70/255 mean luma for our locked params), not
// black, even at a different DPR.
function avgLuma(buf) {
    const png = PNG.sync.read(buf);
    let total = 0;
    for (let i = 0; i < png.data.length; i += 4) {
        total += 0.299 * png.data[i] + 0.587 * png.data[i+1] + 0.114 * png.data[i+2];
    }
    return total / (png.width * png.height);
}

async function withPage({ browser, viewport, deviceScaleFactor, url }, fn) {
    const ctx = await browser.newContext({ viewport, deviceScaleFactor });
    const page = await ctx.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(`[pageerror] ${e.message}`));
    page.on('console', m => { if (m.type() === 'error') errors.push(`[error] ${m.text()}`); });
    await page.goto(url, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1500);
    try {
        return await fn(page);
    } finally {
        if (errors.length) {
            console.error('errors during page run:');
            errors.forEach(l => console.error('  ' + l));
        }
        await ctx.close();
    }
}

async function main() {
    await ensureBuilt();
    await mkdir(OUT, { recursive: true });

    const { server, url } = await startServer();
    const browser = await chromium.launch({
        args: [
            '--enable-unsafe-webgpu',
            '--use-angle=metal',
            '--ignore-gpu-blocklist',
            '--enable-gpu-rasterization',
            '--force-color-profile=srgb',
        ],
    });

    const failures = [];

    // ------------------------------------------------------------------
    // Phase 1 — DPR=2 fills viewport at 2× drawing-buffer
    // ------------------------------------------------------------------
    const VP1 = { width: 800, height: 600 };
    await withPage({
        browser,
        viewport: VP1,
        deviceScaleFactor: 2,
        url: `${url}?t=2.5&mx=0.5&my=0.5`,
    }, async (page) => {
        const s = await readCanvasState(page);
        console.log('[dpr=2]', s);
        if (!s) { failures.push('canvas missing under DPR=2'); return; }
        if (s.dpr !== 2) failures.push(`expected window.devicePixelRatio=2, got ${s.dpr}`);
        if (s.buffer.w !== VP1.width * 2 || s.buffer.h !== VP1.height * 2) {
            failures.push(
                `[dpr=2] buffer ${s.buffer.w}×${s.buffer.h} ≠ viewport×DPR ` +
                `${VP1.width * 2}×${VP1.height * 2}`,
            );
        }
        if (s.css.w !== VP1.width || s.css.h !== VP1.height) {
            failures.push(
                `[dpr=2] CSS size ${s.css.w}×${s.css.h} ≠ viewport ` +
                `${VP1.width}×${VP1.height} — quarter-screen bug?`,
            );
        }

        const png = await page.locator('canvas#wgpu').screenshot({
            path: join(OUT, 'responsive-dpr2.png'),
        });
        const luma = avgLuma(png);
        console.log(`[dpr=2] canvas avgLuma = ${luma.toFixed(2)}/255`);
        if (luma < 8) failures.push(`[dpr=2] canvas appears black (avgLuma ${luma.toFixed(2)})`);
    });

    // ------------------------------------------------------------------
    // Phase 2 — viewport resize updates canvas drawing buffer
    // ------------------------------------------------------------------
    const VP_BEFORE = { width: 800, height: 600 };
    const VP_AFTER  = { width: 1200, height: 900 };
    await withPage({
        browser,
        viewport: VP_BEFORE,
        deviceScaleFactor: 1,
        url: `${url}?t=2.5&mx=0.5&my=0.5`,
    }, async (page) => {
        const before = await readCanvasState(page);
        console.log('[resize] before', before);
        if (before.buffer.w !== VP_BEFORE.width || before.buffer.h !== VP_BEFORE.height) {
            failures.push(
                `[resize] initial buffer ${before.buffer.w}×${before.buffer.h} ≠ ` +
                `${VP_BEFORE.width}×${VP_BEFORE.height}`,
            );
        }

        await page.setViewportSize(VP_AFTER);
        // ResizeObserver fires on the next layout pass; give it a beat plus
        // a frame for State::resize to reconfigure the surface.
        await page.waitForTimeout(600);

        const after = await readCanvasState(page);
        console.log('[resize] after ', after);
        if (after.buffer.w !== VP_AFTER.width || after.buffer.h !== VP_AFTER.height) {
            failures.push(
                `[resize] post-resize buffer ${after.buffer.w}×${after.buffer.h} ≠ ` +
                `${VP_AFTER.width}×${VP_AFTER.height} — winit ResizeObserver path broken?`,
            );
        }
        if (after.css.w !== VP_AFTER.width || after.css.h !== VP_AFTER.height) {
            failures.push(
                `[resize] post-resize CSS ${after.css.w}×${after.css.h} ≠ ` +
                `${VP_AFTER.width}×${VP_AFTER.height}`,
            );
        }

        // The mandelbrot should still render after a resize.
        const png = await page.locator('canvas#wgpu').screenshot({
            path: join(OUT, 'responsive-resize.png'),
        });
        const luma = avgLuma(png);
        console.log(`[resize] post-resize avgLuma = ${luma.toFixed(2)}/255`);
        if (luma < 8) failures.push(`[resize] canvas appears black post-resize (avgLuma ${luma.toFixed(2)})`);
    });

    await browser.close();
    server.close();

    if (failures.length) {
        console.error('\n✗ FAIL:');
        failures.forEach(f => console.error('  - ' + f));
        process.exit(1);
    }
    console.log('\n✓ responsive checks passed');
}

main().catch(e => {
    console.error('test infra error:', e);
    process.exit(2);
});
