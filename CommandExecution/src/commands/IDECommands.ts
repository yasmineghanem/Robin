import * as vscode from 'vscode';
import { FOCUS_TERMINAL, GO_TO_FILE, GO_TO_LINE, KILL_TERMINAL, NEW_TERMINAL, UNDO, REDO, PASTE, CUT, COPY, SELECT_KERNEL, RUN_NOTEBOOK_CELL, RUN_NOTEBOOK, RUN_PYTHON, SELECT, SELECT_RANGE, FIND, GET_ERROR_LIST } from '../constants/IDE';
import fs from "fs";
import { NO_ACTIVE_TEXT_EDITOR } from '../constants/code';
import { get } from 'axios';


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

//select line
const select = () => vscode.commands.registerCommand(SELECT, (data) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
        // const start = new vscode.Position(data.start.line, data.start.character);
        // const end = new vscode.Position(data.end.line, data.end.character);
        // editor.selection = new vscode.Selection(start, end);
        // editor.revealRange(new vscode.Range(start, end));
        vscode.commands.executeCommand(GO_TO_LINE, (data));
        vscode.commands.executeCommand('expandLineSelection');
        return {
            success: true
        };
    } else {
        return {
            success: false,
            message: "No active text editor"
        };

    }
}
);

//select multiple lines
const selectRange = () => vscode.commands.registerCommand(SELECT_RANGE, (data) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
        const start = new vscode.Position(data.startLine ?? 0, data.startCharacter ?? 0);
        const end = new vscode.Position(data.endLine ?? 0, data.endCharacter?? 0);
        editor.selection = new vscode.Selection(start, end);
        editor.revealRange(new vscode.Range(start, end));

        return {
            success: true
        };
    } else {
        return {
            success: false,
            message: "No active text editor"
        };

    }
}
);

// Find
const find = () => vscode.commands.registerCommand(FIND, () => {
    vscode.commands.executeCommand('editor.action.showfind');
}); 

//paste
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

//undo
const undo = () => vscode.commands.registerCommand(UNDO, () => {
    // const editor = vscode.window.activeTextEditor;
    // if (editor) {
    vscode.commands.executeCommand('undo');
    // }
});

//redo
const redo = () => vscode.commands.registerCommand(REDO, () => {
    // const editor = vscode.window.activeTextEditor;
    // if (editor) {
    vscode.commands.executeCommand('redo');
    // }
});

// get error list
const getErrorList = () => vscode.commands.registerCommand(GET_ERROR_LIST, () => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
        const uri = editor.document.uri;
        const errors = vscode.languages.getDiagnostics(uri);
        if (errors.length === 0) {
            return {
                success: true,
                message: "No errors found"
            };
        }
        else{
            const errorList = errors.map((errors) => {
                        return {
                          message: errors.message,
                          line: errors.range.start.line + 1,
                        };
                      });
            return {
                success: true,
                message: errorList

        }

        
        };
    } else {
        return {
            success: false,
            message: "No active text editor"
        };

    }
});


//select kernel for notebook
const selectKernel = () => vscode.commands.registerCommand(SELECT_KERNEL, (data) => {
    const path = `${vscode.workspace.rootPath}\\${data?.path}`;
    const file = vscode.Uri.file(path);

    // check if file exists
    if (fs.existsSync(path)) {
        vscode.workspace.openTextDocument(file).then(doc => {
            vscode.window.showTextDocument(doc);
            //select kernel with data.kernelInfo
            vscode.commands.executeCommand('notebook.selectKernel', {
                kernelInfo: data?.kernelInfo
            });

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
        message: "Selected Kernel does not exist"
    };
});


//run notebook cell
const runNotebookCell = () => vscode.commands.registerCommand(RUN_NOTEBOOK_CELL, () => {
    vscode.commands.executeCommand('notebook.runactivecell');
});

//run all notebook cells
const runNotebook = () => vscode.commands.registerCommand(RUN_NOTEBOOK, () => {
    vscode.commands.executeCommand('notebook.execute');
});

//run python code
const runPython = () => vscode.commands.registerCommand(RUN_PYTHON, (data) => {
    // run current active file from terminal
    const terminal = vscode.window.activeTerminal ?? vscode.window.createTerminal();
    terminal.show();

    // get the current file path
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active text editor.');
        return {
            success: false,
            message: NO_ACTIVE_TEXT_EDITOR
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
        find,
        getErrorList

    ];

    commands.
    forEach(command => command());
};

export default registerIDECommands;