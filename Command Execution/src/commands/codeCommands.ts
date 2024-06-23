import * as vscode from "vscode";
import {
    DECLARE_FUNCTION,
    DECLARE_VARIABLE,
    FILE_EXT_FAILURE,
    FUNCTION_FAILURE,
    FUNCTION_SUCCESS,
    GET_AST,
    NO_ACTIVE_TEXT_EDITOR,
    VARIABLE_FAILURE,
    VARIABLE_SUCCESS,
} from "../constants/code";
import { PythonCodeGenerator } from "../code generation/pythonCodeGenerator";
import { showError, showMessage } from "../communication/utilities";

// utilities

const getCurrentPosition = (editor: vscode.TextEditor): vscode.Position => {
    const position = editor.selection.active;
    return position;
};

const getFileExtension = (editor: vscode.TextEditor): string | undefined => {
    return editor.document.fileName.split(".").pop();
};


// interfaces
interface Parameter {
    name: string;
    value?: any;
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
            // check for extension
            const ext = getFileExtension(editor);

            let codeGenerator;

            switch (ext) {
                case "py":
                    codeGenerator = new PythonCodeGenerator();
                    break;
                default:
                    showError(FILE_EXT_FAILURE);
                    return {
                        success: false,
                        message: FILE_EXT_FAILURE,
                    };
            }

            let s = await editor.edit((editBuilder) => {
                editBuilder.insert(
                    getCurrentPosition(editor),
                    codeGenerator.declareVariable(args.name, args.type, args.value)
                );
            });


            if (!s) {
                showError(VARIABLE_FAILURE);
                return {
                    success: false,
                    message: VARIABLE_FAILURE,
                };
            }
            showMessage(VARIABLE_SUCCESS);
            return {
                success: true,
                message: VARIABLE_SUCCESS,
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

            // check for extension
            const ext = getFileExtension(editor);

            let codeGenerator;

            switch (ext) {
                case "py":
                    codeGenerator = new PythonCodeGenerator();
                    break;
                default:
                    showError(FILE_EXT_FAILURE);
                    return {
                        success: false,
                        message: FILE_EXT_FAILURE,
                    };
            }

            try {
                const code = codeGenerator.declareFunction(args.name, args.parameters ?? []);
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor),
                        code
                    );
                });

                if (!s) {
                    showError(FUNCTION_FAILURE);
                    return {
                        success: false,
                        message: FUNCTION_FAILURE,
                    };
                }

                showMessage(FUNCTION_SUCCESS);
                return {
                    success: true,
                    message: FUNCTION_SUCCESS,
                };



            } catch (e) {
                showError(FUNCTION_FAILURE);
                return {
                    success: false,
                    message: FUNCTION_FAILURE,
                };
            }

        }
        // else {
        //     showError("No active text editor.");
        //     return {
        //         success: false,
        //         message: "No active text editor",
        //     };
        // }
    });
};

const getAST = () => {
    vscode.commands.registerCommand(GET_AST, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            const text = editor.document.getText();

            // get the corresponding AST from the microservice
            const response = await fetch("http://localhost:2806/ast", {
                method: "POST",
                body: JSON.stringify({ code: text }),
                headers: {
                    "Content-Type": "application/json",
                },
            });

            const data: any = await response.json();
            // console.log();
            console.log("ROBIN FILTER \n" + data);

            return {
                success: true,
                message: "AST retrieved successfully",
                data: data,
            };
        }
    });
};

const registerCodeCommands = () => {
    const commands = [declareVariable, declareFunction, getAST];

    commands.forEach((command) => {
        command();
    });
};

export default registerCodeCommands;
