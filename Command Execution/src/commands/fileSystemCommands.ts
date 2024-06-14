import * as vscode from 'vscode';
import fs from "fs";

// create new file
// sample input => 
// {
//     "fileName": "new file",
//     "extension": "/py",
//     "content": "file content"
// }
const createFile = () => vscode.commands.registerCommand('robin.createFile', (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
        const fileName = args.fileName;
        const extension = args.extension;
        const content = args.content;
        const path = `${vscode.workspace.rootPath}\\${fileName}${extension}`;
        // check if it already exists
        if (!fs.existsSync(path)) {
            fs.writeFileSync(path, content);
            return {
                // path,
                success: true
            };
        }
        return {
            // path,
            success: false
        };
    } else {
        vscode.window.showErrorMessage('No active text editor.');
    }
});

// create new directory
// {
//     "name": "new directory"
// }
const createDirectory = () => vscode.commands.registerCommand('robin.createDirectory', (args) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
        const name = args.name;
        const path = `${vscode.workspace.rootPath}\\${name}`;
        // check if it already exists
        if (!fs.existsSync(path)) {
            fs.mkdirSync(path);
            return {
                path,
                success: true
            };
        }
        return {
            path,
            success: false
        };
    } else {
        vscode.window.showErrorMessage('No active text editor.');
    }
});

const registerFileSystemCommands = () => {
    const commands = [
        createFile,
        createDirectory
    ];

    commands.forEach(command => command());
};

export default registerFileSystemCommands;
