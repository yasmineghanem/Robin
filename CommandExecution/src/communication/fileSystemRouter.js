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
const fileSystem_1 = require("../constants/fileSystem");
const router = require('express').Router();
router.post("/create-file", (req, res) => {
    const data = req.body;
    vscode.commands.executeCommand(fileSystem_1.CREATE_FILE, data).then((response) => {
        if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "File created successfully!" }));
        }
        else {
            res.writeHead(409, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "File already exists!" }));
        }
    }, (err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err }));
    });
});
router.post("/create-directory", (req, res) => {
    const data = req.body;
    vscode.commands.executeCommand(fileSystem_1.CREATE_DIRECTORY, data).then((response) => {
        if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({
                // response.path,
                message: "Directory created successfully!",
            }));
        }
        else {
            res.writeHead(409, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "Directory already exists!" }));
        }
    }, (err) => {
        res.writeHead(400, { "Content-Type": "application/json" });
        console.log(err);
        res.end(JSON.stringify(err));
    });
});
// copy file
router.post("/copy-file", (req, res) => {
    const data = req.body;
    vscode.commands.executeCommand(fileSystem_1.COPY_FILE, data).then((response) => {
        if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "File copied successfully!" }));
        }
        else {
            res.writeHead(404, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "File not found" }));
        }
    }, (err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err }));
    });
});
// copy directory
router.post("/copy-directory", (req, res) => {
    const data = req.body;
    vscode.commands.executeCommand(fileSystem_1.COPY_DIRECTORY, data).then((response) => {
        if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "Directory copied successfully!" }));
        }
        else {
            res.writeHead(404, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "Directory not found" }));
        }
    }, (err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err }));
    });
});
// delete file or directory
router.post("/delete", (req, res) => {
    const data = req.body;
    vscode.commands.executeCommand(fileSystem_1.DELETE_FILE, data).then((response) => {
        if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "File/Directory deleted successfully!" }));
        }
        else {
            res.writeHead(404, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "File/Directory not found" }));
        }
    }, (err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err }));
    });
});
// rename file or directory
router.post("/rename", (req, res) => {
    const data = req.body;
    vscode.commands.executeCommand(fileSystem_1.RENAME, data).then((response) => {
        if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "File/Directory renamed successfully!" }));
        }
        else {
            res.writeHead(404, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "File/Directory not found" }));
        }
    }, (err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err }));
    });
});
// copy to clipboard
// router.post(
//   "/copy-clipboard",
//   (req: any, res: any) => {
//     const data = req.body;
//     vscode.commands.executeCommand("robin.copyFileClipboard", data).then(
//       (response: any) => {
//         if (response.success) {
//           res.writeHead(200, { "Content-Type": "application/json" });
//           res.end(JSON.stringify({ message: "File copied to clipboard!" }));
//         }
//         else {
//           res.writeHead(404, { "Content-Type": "application/json" });
//           res.end(JSON.stringify({ message: "File not found" }));
//         }
//       },
//       (err) => {
//         res.writeHead(500, { "Content-Type": "application/json" });
//         res.end(JSON.stringify({ error: err }));
//       }
//     );
//   }
// );
// save
router.post("/save", (req, res) => {
    const data = req.body;
    vscode.commands.executeCommand(fileSystem_1.SAVE, data).then((response) => {
        if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "File saved successfully!" }));
        }
        else {
            res.writeHead(404, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "No active files found" }));
        }
    }, (err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err }));
    });
});
// get files in workspace
router.post("/get-files", (req, res) => {
    // const data = req.body;
    vscode.commands.executeCommand(fileSystem_1.GET_FILES).then((response) => {
        if (response.success) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "Retrieved all files successfully in the workspace!", files: response.files }));
        }
        else {
            res.writeHead(404, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ message: "No files found" }));
        }
    }, (err) => {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err }));
    });
});
exports.default = router;
//# sourceMappingURL=fileSystemRouter.js.map