import * as vscode from 'vscode';
import { DECLARE_FUNCTION, DECLARE_VARIABLE } from '../constants/code';

// utilities
const pythonReservedKeywords: Set<string> = new Set([
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in",
    "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield"
]);

const getCurrentPosition = (editor: vscode.TextEditor): vscode.Position => {
    const position = editor.selection.active;
    return position;
};

function isValidVariableName(name: string): boolean {
    const pattern = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
    if (!pattern.test(name)) {
        return false;
    }
    return !pythonReservedKeywords.has(name);
}

// interfaces
interface Parameter {
    name: string;
    value?: string;
}



// {
//     "name": "x",
//     "value" : 0
// }
// declare variable
const declareVariable = () => {
    vscode.commands.registerCommand(DECLARE_VARIABLE, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            const name = args.name;
            const value = args.value ?? "None";
            if (!isValidVariableName(name)) {
                vscode.window.showErrorMessage("Invalid variable name");
                return {
                    success: false,
                    message: "Invalid variable name"
                };

            }
            const line = `\n${name} = ${JSON.stringify(value)}`;

            let s = await editor.edit(editBuilder => {
                editBuilder.insert(getCurrentPosition(editor), line);
            });


            if (!s) {
                vscode.window.showErrorMessage("Failed to declare variable");
                return {
                    success: false,
                    message: "Failed to declare variable"
                };

            }
            vscode.window.showInformationMessage("Variable declared successfully");
            return {
                success: true,
                message: "Variable declared successfully"
            };


        } else {
            vscode.window.showErrorMessage('No active text editor.');
            return {
                success: false,
                message: "No active text editor"
            };
        }
    });
};


// declare function
// {
//     "name": "x_string",
//     "paramaters": [
//         {
//             "name": "x",
//             "value": "test"
//         },
//         {
//             "name": "y",
//             "value": "test"
//         }
//     ]
// }

const declareFunction = () => {

    vscode.commands.registerCommand(DECLARE_FUNCTION, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // console.log(args)
            const name = args.name;

            if (!isValidVariableName(name)) {
                vscode.window.showErrorMessage("Invalid function name");
                return {
                    success: false,
                    message: "Invalid function name"
                };
            }

            const parameters = args.parameters ?? [];
            // sort the parameters so that the parameters with values are last
            parameters.sort((a: Parameter, b: Parameter) => {
                if ('value' in a && !('value' in b)) {
                    return 1; // a has value, b does not -> a comes after b
                } else if (!('value' in a) && 'value' in b) {
                    return -1; // a does not have value, b does -> a comes before b
                }
                return 0; // both have value or both do not have value -> keep original order
            });

            // console.log(parameters, parameters.length)
            let line1 = "\ndef " + name + "(";
            for (let i = 0; i < parameters.length; i++) {
                if (!isValidVariableName(parameters[i].name)) {
                    return {
                        success: false,
                        message: "Invalid variable name"
                    };
                }

                if (i) { line1 += ", "; }
                line1 += parameters[i]['name'];

                if (parameters[i]['value']) {
                    line1 += ' = ' + JSON.stringify(parameters[i]['value']);

                }
            }
            line1 += "):\n\t";
            let s = await editor.edit(editBuilder => {
                editBuilder.insert(getCurrentPosition(editor), line1);
            });

            if (!s) {
                vscode.window.showErrorMessage("Failed to declare function");
                return {
                    success: false,
                    message: "Failed to declare function"
                };

            }
            vscode.window.showInformationMessage("Function declared successfully");
            return {
                success: true,
                message: "Function declared successfully"
            };
        }
        else {
            vscode.window.showErrorMessage('No active text editor.');
            return {
                success: false,
                message: "No active text editor"
            };
        }

    });
};

const registerCodeCommands = () => {
    const commands = [
        declareVariable,
        declareFunction

    ];

    commands.forEach((command) => {
        command();
    });
};

export default registerCodeCommands;
