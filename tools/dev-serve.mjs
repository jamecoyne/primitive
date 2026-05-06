// `npm run dev:web` — build the wasm bundle, serve web/dist on localhost,
// open the default browser, and keep running until Ctrl+C.
//
// Override the port with PORT=8001 npm run dev:web; falls through to other
// ports automatically if the requested one is in use.

import { spawn, spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(import.meta.url), '../..');
const DIST = join(ROOT, 'web/dist');
const PREFERRED_PORT = Number(process.env.PORT) || 8000;
// If the preferred port is taken, try the next 9 in sequence before giving up.
const PORT_RANGE = 10;

const MIME = {
    '.html': 'text/html; charset=utf-8',
    '.js':   'application/javascript; charset=utf-8',
    '.wasm': 'application/wasm',
    '.css':  'text/css; charset=utf-8',
    '.png':  'image/png',
    '.json': 'application/json; charset=utf-8',
    '.svg':  'image/svg+xml',
    '.ico':  'image/x-icon',
};

function build() {
    console.log('▶ building wasm…');
    const r = spawnSync('./build-web.sh', {
        stdio: 'inherit',
        cwd: ROOT,
    });
    if (r.status !== 0) {
        console.error(`build-web.sh exited with status ${r.status}`);
        process.exit(r.status ?? 1);
    }
}

function makeServer() {
    return createServer(async (req, res) => {
        let urlPath = decodeURIComponent(new URL(req.url, 'http://x').pathname);
        if (urlPath === '/') urlPath = '/index.html';
        const filePath = join(DIST, urlPath);
        if (!filePath.startsWith(DIST)) {
            res.writeHead(403).end('forbidden');
            return;
        }
        try {
            const buf = await readFile(filePath);
            res.writeHead(200, {
                'Content-Type': MIME[extname(filePath)] ?? 'application/octet-stream',
                'Cache-Control': 'no-store',
            });
            res.end(buf);
        } catch {
            res.writeHead(404).end('not found');
        }
    });
}

async function listenOnFirstFreePort(server, start, range) {
    for (let p = start; p < start + range; p++) {
        try {
            await new Promise((resolveListen, rejectListen) => {
                const onError = (err) => {
                    server.removeListener('listening', onListening);
                    rejectListen(err);
                };
                const onListening = () => {
                    server.removeListener('error', onError);
                    resolveListen();
                };
                server.once('error', onError);
                server.once('listening', onListening);
                server.listen(p, '127.0.0.1');
            });
            return p;
        } catch (err) {
            if (err.code !== 'EADDRINUSE') throw err;
            // try the next port
        }
    }
    throw new Error(`no free port in range ${start}..${start + range - 1}`);
}

function openBrowser(url) {
    const opener = process.platform === 'darwin'
        ? 'open'
        : process.platform === 'win32'
            ? 'cmd'
            : 'xdg-open';
    const args = process.platform === 'win32' ? ['/c', 'start', '', url] : [url];
    try {
        spawn(opener, args, { stdio: 'ignore', detached: true }).unref();
    } catch (err) {
        console.warn(`couldn't open browser automatically: ${err.message}`);
        console.warn(`open this URL manually: ${url}`);
    }
}

async function main() {
    try {
        await stat(ROOT);
    } catch {
        console.error(`project root missing: ${ROOT}`);
        process.exit(2);
    }

    build();

    const server = makeServer();
    const port = await listenOnFirstFreePort(server, PREFERRED_PORT, PORT_RANGE);
    const url = `http://127.0.0.1:${port}/`;

    console.log(`▶ serving ${DIST} at ${url}`);
    console.log('▶ press Ctrl+C to stop');

    openBrowser(url);

    const shutdown = (signal) => {
        console.log(`\n▶ ${signal} received — shutting down`);
        server.close(() => process.exit(0));
        // Force-exit if connections linger.
        setTimeout(() => process.exit(0), 1000).unref();
    };
    process.on('SIGINT', () => shutdown('SIGINT'));
    process.on('SIGTERM', () => shutdown('SIGTERM'));
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
