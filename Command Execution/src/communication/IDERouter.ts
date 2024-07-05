import * as vscode from 'vscode';
import { FOCUS_TERMINAL, GO_TO_FILE, GO_TO_LINE, KILL_TERMINAL, NEW_TERMINAL, UNDO, REDO,COPY, SELECT_KERNEL, RUN_NOTEBOOK_CELL, RUN_NOTEBOOK, RUN_PYTHON } from '../constants/IDE';
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
        vscode.commands.executeCommand('copy').then(
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
            }
            ,
            (err) => {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify(err));
            }
        );
    }
);

//UNDO
router.get(
    "/undo",
    (req: any, res: any) => {
        vscode.commands.executeCommand('undo').then(
            () => {
                res.writeHead(200, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ message: "Undo Done!" }));
            },
            (err) => {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify(err));
            }
        );
    }
);

//REDO
router.get(
    "/redo",
    (req: any, res: any) => {
        vscode.commands.executeCommand('redo').then(
            () => {
                res.writeHead(200, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ message: "Redo Done!" }));
            },
            (err) => {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify(err));
            }
        );
    }
);

// select kernel
router.post(
    "/select-kernel",
    (req: any, res: any) => {
        const data = req.body;
        vscode.commands.executeCommand(SELECT_KERNEL, data).then(
            () => {
                res.writeHead(200, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ message: "Kernel selected!" }));
            },
            (err) => {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify(err));
            }
        );
    }
);

// run notebook cell
router.get(
    "/run-notebook-cell",
    (req: any, res: any) => {
        vscode.commands.executeCommand(RUN_NOTEBOOK_CELL).then(
            () => {
                res.writeHead(200, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ message: "Notebook cell run!" }));
            },
            (err) => {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify(err));
            }
        );
    }
);

// run all notebook cells
router.get(
    "/run-notebook",
    (req: any, res: any) => {
        vscode.commands.executeCommand(RUN_NOTEBOOK).then(
            () => {
                res.writeHead(200, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ message: "Notebook run!" }));
            },
            (err) => {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify(err));
            }
        );
    }
);

//run python file
router.post(
    "/run-python-file",
    (req: any, res: any) => {
        const data = req.body;
        vscode.commands.executeCommand(RUN_PYTHON, data).then(
            () => {
                res.writeHead(200, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ message: "Python file run!" }));
            },
            (err) => {
                res.writeHead(400, { "Content-Type": "application/json" });
                res.end(JSON.stringify(err));
            }
        );
    }
);

export default router;