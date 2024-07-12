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
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const vscode = __importStar(require("vscode"));
const IDE_1 = require("../constants/IDE");
const fs_1 = __importDefault(require("fs"));
const code_1 = require("../constants/code");
// go to line
const goToLine = () => vscode.commands.registerCommand(IDE_1.GO_TO_LINE, (data) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
        const position = new vscode.Position(data.line - 1 ?? 0, data.character ?? 0);
        editor.selection = new vscode.Selection(position, position);
        editor.revealRange(new vscode.Range(position, position));
        return {
            success: true
        };
    }
    else {
        return {
            success: false,
            message: "No active text editor"
        };
    }
});
// go to file
const goToFile = () => vscode.commands.registerCommand(IDE_1.GO_TO_FILE, (data) => {
    const path = `${vscode.workspace.rootPath}\\${data?.path}`;
    const file = vscode.Uri.file(path);
    // check if file exists
    if (fs_1.default.existsSync(path)) {
        vscode.workspace.openTextDocument(file).then(doc => {
            vscode.window.showTextDocument(doc);
            return {
                success: true
            };
        }, (err) => {
            return {
                success: false,
                message: err
            };
        });
    }
    return {
        success: false,
        message: "File does not exist"
    };
});
// focus on terminal
const focusTerminal = () => vscode.commands.registerCommand(IDE_1.FOCUS_TERMINAL, () => {
    vscode.commands.executeCommand('workbench.action.terminal.focus');
});
// new terminal
const newTerminal = () => vscode.commands.registerCommand(IDE_1.NEW_TERMINAL, () => {
    vscode.commands.executeCommand('workbench.action.terminal.new');
});
// kill terminal
const killTerminal = () => vscode.commands.registerCommand(IDE_1.KILL_TERMINAL, () => {
    vscode.commands.executeCommand('workbench.action.terminal.kill');
});
//select line
const select = () => vscode.commands.registerCommand(IDE_1.SELECT, (data) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
        // const start = new vscode.Position(data.start.line, data.start.character);
        // const end = new vscode.Position(data.end.line, data.end.character);
        // editor.selection = new vscode.Selection(start, end);
        // editor.revealRange(new vscode.Range(start, end));
        vscode.commands.executeCommand(IDE_1.GO_TO_LINE, (data));
        vscode.commands.executeCommand('expandLineSelection');
        return {
            success: true
        };
    }
    else {
        return {
            success: false,
            message: "No active text editor"
        };
    }
});
//select multiple lines
const selectRange = () => vscode.commands.registerCommand(IDE_1.SELECT_RANGE, (data) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
        const start = new vscode.Position(data.startLine ?? 0, data.startCharacter ?? 0);
        const end = new vscode.Position(data.endLine ?? 0, data.endCharacter ?? 0);
        editor.selection = new vscode.Selection(start, end);
        editor.revealRange(new vscode.Range(start, end));
        return {
            success: true
        };
    }
    else {
        return {
            success: false,
            message: "No active text editor"
        };
    }
});
// Find
const find = () => vscode.commands.registerCommand(IDE_1.FIND, () => {
    vscode.commands.executeCommand('editor.action.showfind');
});
//paste
const paste = () => vscode.commands.registerCommand(IDE_1.PASTE, async () => {
    // implement the paste itself
    // check if the cursor is selecting an area, replace it with the clipboard content
    // if not, paste the clipboard content at the current cursor position
    const editor = vscode.window.activeTextEditor;
    if (editor) {
        const selection = editor.selection;
        const text = await vscode.env.clipboard.readText();
        editor.edit(editBuilder => {
            editBuilder.replace(selection, text);
        });
    }
    else {
        vscode.window.showErrorMessage('No active text editor.');
        return {
            success: false,
            message: code_1.NO_ACTIVE_TEXT_EDITOR
        };
    }
});
// cut
const cut = () => vscode.commands.registerCommand(IDE_1.CUT, () => {
    // implement the cut itself
    // check if the cursor is selecting an area, copy it to clipboard and replace it with an empty string
    // if not, copy the current line to clipboard
    const editor = vscode.window.activeTextEditor;
    if (editor) {
        const selection = editor.selection;
        const text = editor.document.getText(selection);
        if (text) {
            vscode.env.clipboard.writeText(text);
            editor.edit(editBuilder => {
                editBuilder.replace(selection, '');
            });
        }
        else {
            const line = editor.document.lineAt(selection.active.line);
            vscode.env.clipboard.writeText(line.text);
            editor.edit(editBuilder => {
                editBuilder.delete(line.range);
            });
        }
    }
    else {
        vscode.window.showErrorMessage('No active text editor.');
        return {
            success: false,
            message: code_1.NO_ACTIVE_TEXT_EDITOR
        };
    }
});
// copy
const copy = () => vscode.commands.registerCommand(IDE_1.COPY, () => vscode.commands.executeCommand('editor.action.clipboardCopyAction'));
//undo
const undo = () => vscode.commands.registerCommand(IDE_1.UNDO, () => {
    // const editor = vscode.window.activeTextEditor;
    // if (editor) {
    vscode.commands.executeCommand('undo');
    // }
});
//redo
const redo = () => vscode.commands.registerCommand(IDE_1.REDO, () => {
    // const editor = vscode.window.activeTextEditor;
    // if (editor) {
    vscode.commands.executeCommand('redo');
    // }
});
//select kernel for notebook
const selectKernel = () => vscode.commands.registerCommand(IDE_1.SELECT_KERNEL, (data) => {
    const path = `${vscode.workspace.rootPath}\\${data?.path}`;
    const file = vscode.Uri.file(path);
    // check if file exists
    if (fs_1.default.existsSync(path)) {
        vscode.workspace.openTextDocument(file).then(doc => {
            vscode.window.showTextDocument(doc);
            //select kernel with data.kernelInfo
            vscode.commands.executeCommand('notebook.selectKernel', {
                kernelInfo: data?.kernelInfo
            });
            return {
                success: true
            };
        }, (err) => {
            return {
                success: false,
                message: err
            };
        });
    }
    return {
        success: false,
        message: "Selected Kernel does not exist"
    };
});
//run notebook cell
const runNotebookCell = () => vscode.commands.registerCommand(IDE_1.RUN_NOTEBOOK_CELL, () => {
    vscode.commands.executeCommand('notebook.runactivecell');
});
//run all notebook cells
const runNotebook = () => vscode.commands.registerCommand(IDE_1.RUN_NOTEBOOK, () => {
    vscode.commands.executeCommand('notebook.execute');
});
//run python code
const runPython = () => vscode.commands.registerCommand(IDE_1.RUN_PYTHON, (data) => {
    // run current active file from terminal
    const terminal = vscode.window.activeTerminal ?? vscode.window.createTerminal();
    terminal.show();
    // get the current file path
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active text editor.');
        return {
            success: false,
            message: code_1.NO_ACTIVE_TEXT_EDITOR
        };
    }
    try { // relative path
        const path = editor.document.fileName;
        // get the file name that ends with .py
        const fileName = path.split('\\').pop();
        if (!fileName?.endsWith('.py')) {
            vscode.window.showErrorMessage('File is not a python file.');
            return {
                success: false,
                message: 'File is not a python file.'
            };
        }
        terminal.sendText(`python ./${fileName}`);
        return {
            success: true,
            message: 'Python file running.'
        };
    }
    catch (err) {
        vscode.window.showErrorMessage('Error running python file.');
        return {
            success: false,
            message: 'Error running python file.'
        };
    }
});
// register commands
const registerIDECommands = () => {
    const commands = [
        goToLine,
        goToFile,
        focusTerminal,
        newTerminal,
        killTerminal,
        paste,
        cut,
        copy,
        undo,
        redo,
        selectKernel,
        runNotebookCell,
        runNotebook,
        runPython,
        select,
        selectRange,
        find
    ];
    commands.
        forEach(command => command());
};
exports.default = registerIDECommands;
//# sourceMappingURL=IDECommands.js.map