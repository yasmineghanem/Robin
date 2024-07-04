import * as vscode from 'vscode';
import { COPY, CUT, FOCUS_TERMINAL, GO_TO_FILE, GO_TO_LINE, KILL_TERMINAL, NEW_TERMINAL, PASTE } from '../constants/IDE';
const router = require('express').Router();




router.post(
    "/go-to-line",
    (req: any, res: any) => {
        const data = req.body;
        vscode.commands.executeCommand(GO_TO_LINE, data).then(
            (response: any) => {
                if (response.success) {
                    res.writeHead(200, { "Content-Type": "application/json" });
                    res.end(JSON.stringify({ message: "Navigated to line!" }));
                } else {
                    res.writeHead(400, { "Content-Type": "application/json" });
                    res.end(JSON.stringify({ message: "No active text editor" }));
                }
            },
            (err) => {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify(err));
            }
        );
    }
);

// go to file
router.post(
    "/go-to-file",
    (req: any, res: any) => {
        const data = req.body;
        vscode.commands.executeCommand(GO_TO_FILE, data).then(
            (response: any) => {
                if (response.success) {
                    res.writeHead(200, { "Content-Type": "application/json" });
                    res.end(JSON.stringify({ message: "Navigated to file!" }));
                } else {
                    res.writeHead(400, { "Content-Type": "application/json" });
                    res.end(JSON.stringify({ message: response.message }));
                }
            },
            (err) => {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify(err));
            }
        );
    }
);


// terminal focus
router.get(
    "/focus-terminal",
    (req: any, res: any) => {
        vscode.commands.executeCommand(FOCUS_TERMINAL).then(
            () => {
                res.writeHead(200, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ message: "Terminal focused!" }));
            },
            (err) => {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify(err));
            }
        );
    }
);

// new terminal
router.get(
    "/new-terminal",
    (req: any, res: any) => {
        vscode.commands.executeCommand(NEW_TERMINAL).then(
            () => {
                res.writeHead(200, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ message: "New terminal created!" }));
            },
            (err) => {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify(err));
            }
        );
    }
);

// kill terminal
router.get(
    "/kill-terminal",
    (req: any, res: any) => {
        vscode.commands.executeCommand(KILL_TERMINAL).then(
            () => {
                res.writeHead(200, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ message: "Terminal killed!" }));
            },
            (err) => {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify(err));
            }
        );
    }
);

// copy
router.get(
    "/copy",
    (req: any, res: any) => {
        vscode.commands.executeCommand(COPY).then(
            () => {
                res.writeHead(200, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ message: "Copied!" }));
            },
            (err) => {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify(err));
            }
        );
    }
);
router.get(
    "/paste",
    (req: any, res: any) => {
        vscode.commands.executeCommand('paste').then(
            () => {
                res.writeHead(200, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ message: "Pasted!" }));
            },
            (err) => {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify(err));
            }
        );
    }
);
router.get(
    "/cut",
    (req: any, res: any) => {
        vscode.commands.executeCommand('cut').then(
            () => {
                res.writeHead(200, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ message: "Cut!" }));
            },
            (err) => {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify(err));
            }
        );
    }
);



export default router;