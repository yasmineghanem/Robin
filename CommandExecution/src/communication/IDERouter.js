"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
const vscode = __importStar(require("vscode"));
const IDE_1 = require("../constants/IDE");
const router = require('express').Router();
router.post("/go-to-line", (req, res) => {
    const data = req.body;
    vscode.commands.executeCommand(IDE_1.GO_TO_LINE, data).then((response) => {
        if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "Navigated to line!" }));
        }
        else {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "No active text editor" }));
        }
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
// go to file
router.post("/go-to-file", (req, res) => {
    const data = req.body;
    vscode.commands.executeCommand(IDE_1.GO_TO_FILE, data).then((response) => {
        if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "Navigated to file!" }));
        }
        else {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: response.message }));
        }
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
// terminal focus
router.get("/focus-terminal", (req, res) => {
    vscode.commands.executeCommand(IDE_1.FOCUS_TERMINAL).then(() => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Terminal focused!" }));
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
// new terminal
router.get("/new-terminal", (req, res) => {
    vscode.commands.executeCommand(IDE_1.NEW_TERMINAL).then(() => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "New terminal created!" }));
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
// kill terminal
router.get("/kill-terminal", (req, res) => {
    vscode.commands.executeCommand(IDE_1.KILL_TERMINAL).then(() => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Terminal killed!" }));
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
// copy
router.get("/copy", (req, res) => {
    vscode.commands.executeCommand('copy').then(() => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Copied!" }));
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
router.get("/paste", (req, res) => {
    vscode.commands.executeCommand('paste').then(() => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Pasted!" }));
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
router.get("/cut", (req, res) => {
    vscode.commands.executeCommand('cut').then(() => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Cut!" }));
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
//UNDO
router.get("/undo", (req, res) => {
    vscode.commands.executeCommand('undo').then(() => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Undo Done!" }));
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
//REDO
router.get("/redo", (req, res) => {
    vscode.commands.executeCommand('redo').then(() => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Redo Done!" }));
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
// select kernel
router.post("/select-kernel", (req, res) => {
    const data = req.body;
    vscode.commands.executeCommand(IDE_1.SELECT_KERNEL, data).then(() => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Kernel selected!" }));
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
// run notebook cell
router.get("/run-notebook-cell", (req, res) => {
    vscode.commands.executeCommand(IDE_1.RUN_NOTEBOOK_CELL).then(() => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Notebook cell run!" }));
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
// run all notebook cells
router.get("/run-notebook", (req, res) => {
    vscode.commands.executeCommand(IDE_1.RUN_NOTEBOOK).then(() => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Notebook run!" }));
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
//run python file
router.post("/run-python-file", (req, res) => {
    const data = req.body;
    vscode.commands.executeCommand(IDE_1.RUN_PYTHON, data).then(() => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Python file run!" }));
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
//select
router.get("/select", (req, res) => {
    const data = req.body;
    // vscode.commands.executeCommand('editor.action.smartSelect.expand', data).then(
    vscode.commands.executeCommand(IDE_1.SELECT, data).then(() => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Selected!" }));
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
//select range
router.get("/select-range", (req, res) => {
    const data = req.body;
    // vscode.commands.executeCommand('editor.action.smartSelect', data).then(
    vscode.commands.executeCommand(IDE_1.SELECT_RANGE, data).then(() => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Selected Range!" }));
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
//find
router.get("/find", (req, res) => {
    vscode.commands.executeCommand(IDE_1.FIND).then(() => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ message: "Find!" }));
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify(err));
    });
});
exports.default = router;
//# sourceMappingURL=IDERouter.js.map