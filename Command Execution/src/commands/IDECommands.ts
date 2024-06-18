import * as vscode from 'vscode';
import { FOCUS_TERMINAL, GO_TO_FILE, GO_TO_LINE, KILL_TERMINAL, NEW_TERMINAL } from '../constants/IDE';
import fs from "fs";


// go to line
const goToLine = () => vscode.commands.registerCommand(GO_TO_LINE, (data) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
        const position = new vscode.Position(data.line - 1 ?? 0, data.character ?? 0);
        editor.selection = new vscode.Selection(position, position);
        editor.revealRange(new vscode.Range(position, position));

        return {
            success: true
        };
    } else {
        return {
            success: false,
            message: "No active text editor"
        };

    }
});

// go to file
const goToFile = () => vscode.commands.registerCommand(GO_TO_FILE, (data) => {

    const path = `${vscode.workspace.rootPath}\\${data?.path}`;
    const file = vscode.Uri.file(path);

    // check if file exists
    if (fs.existsSync(path)) {
        vscode.workspace.openTextDocument(file).then(doc => {
            vscode.window.showTextDocument(doc);
            return {
                success: true
            };
        },
            (err) => {
                return {
                    success: false,
                    message: err
                };
            }
        );

    }

    return {
        success: false,
        message: "File does not exist"
    };


});


// focus on terminal
const focusTerminal = () => vscode.commands.registerCommand(FOCUS_TERMINAL, () => {
    vscode.commands.executeCommand('workbench.action.terminal.focus');
});

// new terminal
const newTerminal = () => vscode.commands.registerCommand(NEW_TERMINAL, () => {
    vscode.commands.executeCommand('workbench.action.terminal.new');
});

// kill terminal
const killTerminal = () => vscode.commands.registerCommand(KILL_TERMINAL, () => {
    vscode.commands.executeCommand('workbench.action.terminal.kill');
});
// register commands
const registerIDECommands = () => {
    const commands = [
        goToLine,
        goToFile,
        focusTerminal,
        newTerminal,
        killTerminal
    ];

    commands.forEach(command => command());
};

export default registerIDECommands;