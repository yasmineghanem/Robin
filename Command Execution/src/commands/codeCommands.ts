import * as vscode from "vscode";
import {
    ADD_WHITESPACE,
    DECLARE_CONSTANT,
    DECLARE_FUNCTION,
    DECLARE_VARIABLE,
    ASSIGN_VARIABLE,
    FILE_EXT_FAILURE,
    FUNCTION_CALL,
    FUNCTION_CALL_FAILURE,
    FUNCTION_CALL_SUCCESS,
    FUNCTION_FAILURE,
    FUNCTION_SUCCESS,
    GET_AST,
    NO_ACTIVE_TEXT_EDITOR,
    VARIABLE_FAILURE,
    VARIABLE_SUCCESS,
    WHITE_SPACE_FAILURE,
    WHITE_SPACE_SUCCESS,
    ASSIGNMENT_FAILURE,
    ASSIGNMENT_SUCCESS,
} from "../constants/code";
import { PythonCodeGenerator } from "../code generation/pythonCodeGenerator";
import { showError, showMessage } from "../communication/utilities";
import { EXTENSIONS } from "../constants/constants";

// utilities

const getCurrentPosition = (editor: vscode.TextEditor): vscode.Position => {
    const position = editor.selection.active;
    return position;
};

const getFileExtension = (editor: vscode.TextEditor): string | undefined => {
    return editor.document.fileName.split(".").pop();
};

const handleFailure = (message: string): any => {
    showError(message);
    return {
        success: false,
        message: message,
    };
};

const handleSuccess = (message: string): any => {
    showMessage(message);
    return {
        success: true,
        message: message,
    };
};

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
                case EXTENSIONS.PYTHON:
                    codeGenerator = new PythonCodeGenerator();
                    break;
                case EXTENSIONS.JUPYTER:
                    codeGenerator = new PythonCodeGenerator();
                    break;
                default:
                    return handleFailure(FILE_EXT_FAILURE);
            }

            let s = await editor.edit((editBuilder) => {
                editBuilder.insert(
                    getCurrentPosition(editor),
                    codeGenerator.declareVariable(args.name, args.type, args.value)
                );
            });

            if (!s) {
                return handleFailure(VARIABLE_FAILURE);
            }

            return handleSuccess(VARIABLE_SUCCESS);
        }
        return handleFailure(NO_ACTIVE_TEXT_EDITOR);
    });
};

const assignVariable = () => {
    vscode.commands.registerCommand(ASSIGN_VARIABLE, async (args) => {
        const editor = vscode.window.activeTextEditor;

        if (editor) {

            let codeGenerator = new PythonCodeGenerator();
            let s = await editor.edit((editBuilder) => {
                editBuilder.insert(
                    getCurrentPosition(editor),
                    codeGenerator.assignVariable(
                        args.name,
                        args.value,
                        args.type
                    )
                );
            });

            if (!s) {
                return handleFailure(ASSIGNMENT_FAILURE);
            }

            return handleSuccess(ASSIGNMENT_SUCCESS);
        }
    });
};

// declare constant
const declareConstant = () => {
    vscode.commands.registerCommand(DECLARE_CONSTANT, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);

            let codeGenerator;

            switch (ext) {
                case EXTENSIONS.PYTHON:
                    codeGenerator = new PythonCodeGenerator();
                    break;
                case EXTENSIONS.JUPYTER:
                    codeGenerator = new PythonCodeGenerator();
                    break;
                default:
                    return handleFailure(FILE_EXT_FAILURE);
            }

            try {

                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(
                        getCurrentPosition(editor),
                        codeGenerator.declareConstant(args.name, args.value)
                    );
                });

                if (!s) {
                    return handleFailure(VARIABLE_FAILURE);
                }
            }
            catch (e) {
                return handleFailure(VARIABLE_FAILURE);
            }

            return handleSuccess(VARIABLE_SUCCESS);
        }
        return handleFailure(NO_ACTIVE_TEXT_EDITOR);
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
                case EXTENSIONS.PYTHON:
                    codeGenerator = new PythonCodeGenerator();
                    break;
                case EXTENSIONS.JUPYTER:
                    codeGenerator = new PythonCodeGenerator();
                    break;
                default:
                    return handleFailure(FILE_EXT_FAILURE);
            }

            try {
                const code = codeGenerator.declareFunction(
                    args.name,
                    args.parameters ?? []
                );
                let s = await editor.edit((editBuilder) => {
                    editBuilder.insert(getCurrentPosition(editor), code);
                });

                if (!s) {
                    return handleFailure(FUNCTION_FAILURE);
                }

                return handleSuccess(FUNCTION_SUCCESS);
            } catch (e) {
                return handleFailure(FUNCTION_FAILURE);
            }
        }
        return handleFailure(NO_ACTIVE_TEXT_EDITOR);
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

// {
//     "name": "x_string",
//         "args": [
//             {
//                 "name": "x",
//                 "value": "test"
//             },
//             {
//                 "name": "y",
//                 "value": "test"
//             }
//         ]
// }

// function call
const functionCall = () => {
    vscode.commands.registerCommand(FUNCTION_CALL, async (args) => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            // check for extension
            const ext = getFileExtension(editor);

            let codeGenerator;

            switch (ext) {
                case EXTENSIONS.PYTHON:
                    codeGenerator = new PythonCodeGenerator();
                    break;
                case EXTENSIONS.JUPYTER:
                    codeGenerator = new PythonCodeGenerator();
                    break;
                default:
                    return handleFailure(FILE_EXT_FAILURE);
            }

            let s = await editor.edit((editBuilder) => {
                editBuilder.insert(
                    getCurrentPosition(editor),
                    codeGenerator.generateFunctionCall(args.name, args.args)
                );
            });

            if (!s) {
                return handleFailure(FUNCTION_CALL_FAILURE);
            }

            return handleSuccess(FUNCTION_CALL_SUCCESS);
        }
    });
};


// white space
const addWhiteSpace = () => {
    vscode.commands.registerCommand(ADD_WHITESPACE, async (args) => {
        const editor = vscode.window.activeTextEditor;

        if (editor) {

            let codeGenerator = new PythonCodeGenerator();
            let s = await editor.edit((editBuilder) => {
                editBuilder.insert(
                    getCurrentPosition(editor),
                    codeGenerator.addWhiteSpace(
                        args.type,
                        args?.count
                    )
                );
            });

            if (!s) {
                return handleFailure(WHITE_SPACE_FAILURE);
            }

            return handleSuccess(WHITE_SPACE_SUCCESS);
        }
    });
};



const registerCodeCommands = () => {
    const commands = [declareVariable, declareFunction, getAST, functionCall, declareConstant,
        assignVariable,
        addWhiteSpace
    ];

    commands.forEach((command) => {
        command();
    });
};

export default registerCodeCommands;
