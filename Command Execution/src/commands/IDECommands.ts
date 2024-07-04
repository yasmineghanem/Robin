import * as vscode from 'vscode';
import { COPY, CUT, FOCUS_TERMINAL, GO_TO_FILE, GO_TO_LINE, KILL_TERMINAL, NEW_TERMINAL, PASTE } from '../constants/IDE';
import fs from "fs";
import { NO_ACTIVE_TEXT_EDITOR } from '../constants/code';


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


const paste = () => vscode.commands.registerCommand(PASTE,
    async () => {
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
                message: NO_ACTIVE_TEXT_EDITOR
            };
        }

    }
);

// cut
const cut = () => vscode.commands.registerCommand(CUT,
    () => {
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
            } else {
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
                message: NO_ACTIVE_TEXT_EDITOR
            };
        }
    }

);

// copy
const copy = () => vscode.commands.registerCommand(COPY,
    () => vscode.commands.executeCommand('editor.action.clipboardCopyAction')
);

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
        copy
    ];

    commands.forEach(command => command());
};

export default registerIDECommands;