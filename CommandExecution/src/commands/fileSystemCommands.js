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
const fs_1 = __importDefault(require("fs"));
const fileSystem_1 = require("../constants/fileSystem");
/**
 *
 * defining some utilities
 */
// check if the file or directory exists
// if check with and without extension
const fileExists = (path) => {
    if (fs_1.default.existsSync(path)) {
        return true;
    }
    const files = fs_1.default.readdirSync(vscode.workspace.rootPath?.toString() || '');
    const file = files.find(file => file.split('.')[0] === path);
    if (file) {
        return true;
    }
    return false;
};
// get list of all files in workspace directory
const getFiles = () => {
    vscode.commands.registerCommand(fileSystem_1.GET_FILES, () => {
        const files = fs_1.default.readdirSync(vscode.workspace.rootPath?.toString() || '');
        return {
            files,
            success: true
        };
    });
};
// create new file
// sample input => 
// {
//     "fileName": "new file",
//     "extension": "/py",
//     "content": "file content"
// }
const createFile = () => vscode.commands.registerCommand(fileSystem_1.CREATE_FILE, (args) => {
    // const editor = vscode.window.activeTextEditor;
    // if (editor) {
    const fileName = args.fileName;
    const extension = args.extension;
    const content = args.content;
    const path = `${vscode.workspace.rootPath}\\${fileName}${extension}`;
    // check if it already exists
    if (!fs_1.default.existsSync(path)) {
        fs_1.default.writeFileSync(path, content);
        return {
            // path,
            success: true
        };
    }
    return {
        // path,
        success: false
    };
    // } else {
    //     vscode.window.showErrorMessage('No active text editor.');
    // }
});
// create new directory
// {
//     "name": "new directory"
// }
const createDirectory = () => vscode.commands.registerCommand(fileSystem_1.CREATE_DIRECTORY, (args) => {
    // const editor = vscode.window.activeTextEditor;
    //     if (editor) {
    const name = args.name;
    const path = `${vscode.workspace.rootPath}\\${name}`;
    // check if it already exists
    if (!fs_1.default.existsSync(path)) {
        fs_1.default.mkdirSync(path);
        return {
            path,
            success: true
        };
    }
    return {
        path,
        success: false
    };
    // } else {
    //     vscode.window.showErrorMessage('No active text editor.');
    // }
});
/**
 * copy file
 * {
 *   source: "source file",
 *   destination: "destination file"
 * }
 *
 * first, we need to check if the source file exists,
 * if it does, we need to read it's content
 * check if the destination is defined
 * if it is, create a new file with the source file's content, and the destination's name
 * if it's not, create a new file with the source file's content and the name would be source file - copy
 */
const copyFileCommand = () => {
    vscode.commands.registerCommand(fileSystem_1.COPY_FILE, (args) => {
        const source = args.source;
        const destination = args.destination;
        const sourcePath = `${vscode.workspace.rootPath}\\${source}`;
        const destinationPath = `${vscode.workspace.rootPath}\\${destination}`;
        let path = "";
        // check if the source file doesn't have an extension, get the first the file the matches the source
        if (source.split('.').pop() === source) {
            const files = fs_1.default.readdirSync(vscode.workspace.rootPath?.toString() || '');
            const file = files.find(file => file.split('.')[0] === source);
            if (file) {
                const content = fs_1.default.readFileSync(`${vscode.workspace.rootPath}\\${file}`, 'utf-8');
                if (destination) {
                    path = destinationPath;
                    fs_1.default.writeFileSync(destinationPath, content);
                }
                else {
                    // add copy before extenstion
                    path = `${vscode.workspace.rootPath}\\${source}-copy.${file.split('.').pop()}`;
                    fs_1.default.writeFileSync(path, content);
                }
                return {
                    success: true,
                    path
                };
            }
        }
        else {
            if (fs_1.default.existsSync(sourcePath)) {
                const content = fs_1.default.readFileSync(sourcePath, 'utf-8');
                if (destination) {
                    fs_1.default.writeFileSync(destinationPath, content);
                }
                else {
                    // add copy before extension
                    path = `${vscode.workspace.rootPath}\\${source.split('.')[0]}-copy.${source.split('.').pop()}`;
                    fs_1.default.writeFileSync(path, content);
                }
                return {
                    success: true,
                    path
                };
            }
        }
        return {
            success: false,
            message: "Source file not found"
        };
    });
};
const copyDirectory = () => {
    vscode.commands.registerCommand(fileSystem_1.COPY_DIRECTORY, (args) => {
        const source = args.source;
        const destination = args.destination;
        const overwrite = args.overwrite ?? false;
        const sourcePath = `${vscode.workspace.rootPath}\\${source}`;
        if (fs_1.default.existsSync(sourcePath)) {
            if (destination) {
                vscode.workspace.fs.copy(vscode.Uri.file(sourcePath), vscode.Uri.file(`${vscode.workspace.rootPath}\\${destination}`));
            }
            else {
                vscode.workspace.fs.copy(vscode.Uri.file(sourcePath), vscode.Uri.file(`${vscode.workspace.rootPath}\\${source} copy`));
            }
            return {
                success: true
            };
        }
        return {
            success: false,
            message: "Source directory not found"
        };
    });
};
// delete file or directory
const deleteFile = () => {
    vscode.commands.registerCommand(fileSystem_1.DELETE_FILE, (args) => {
        const source = args.source;
        const sourcePath = `${vscode.workspace.rootPath}\\${source}`;
        if (fileExists(sourcePath)) {
            vscode.workspace.fs.delete(vscode.Uri.file(sourcePath), { recursive: true });
            return {
                success: true
            };
        }
        return {
            success: false,
            message: "File not found"
        };
    });
};
// copy file to clipboard
// const copyFileClipboard = () => {
//     vscode.commands.registerCommand('robin.copyFileClipboard', (args) => {
//         const source = args.source;
//         const sourcePath = `${vscode.workspace.rootPath}\\${source}`;
//         if (fileExists(sourcePath)) {
//             vscode.env.clipboard.writeText(sourcePath);
//             return {
//                 success: true
//             };
//         }
//         return {
//             success: false,
//             message: "File not found"
//         };
//     });
// };
// paste from clipboard
// const pasteFromClipboard = () => {
//     vscode.commands.registerCommand('robin.pasteFromClipboard', (args) => {
//         vscode.env.clipboard.readText().then((text) => {
//             if (fileExists(text)) {
//                 vscode.workspace.fs.copy(
//                     vscode.Uri.file(text),
//                     vscode.Uri.file(`${vscode.workspace.rootPath}\\${text.split('\\').pop()}`)
//                 );
//                 return {
//                     success: true
//                 };
//             }
//             return {
//                 success: false,
//                 message: "File not found"
//             };
//         });
//     });
// };
const rename = () => {
    vscode.commands.registerCommand(fileSystem_1.RENAME, (args) => {
        const source = args.source;
        const destination = args.destination;
        // const overwrite = args.overwrite ?? false;
        const sourcePath = `${vscode.workspace.rootPath}\\${source}`;
        const destinationPath = `${vscode.workspace.rootPath}\\${destination}`;
        if (fileExists(sourcePath)) {
            vscode.workspace.fs.rename(vscode.Uri.file(sourcePath), vscode.Uri.file(destinationPath));
            return {
                success: true,
            };
        }
        return {
            success: false,
            message: "File not found"
        };
    });
};
// save file or all current files
const saveFile = () => {
    vscode.commands.registerCommand(fileSystem_1.SAVE, (args) => {
        const all = args.all ?? false;
        if (all) {
            vscode.workspace.textDocuments.forEach((doc) => {
                doc.save();
            });
        }
        else {
            const editor = vscode.window.activeTextEditor;
            if (editor) {
                editor.document.save();
                // return {
                //     success: true
                // };
            }
            ;
        }
        return {
            success: true
        };
    });
};
const registerFileSystemCommands = () => {
    const commands = [
        createFile,
        createDirectory,
        copyFileCommand,
        copyDirectory,
        deleteFile,
        rename,
        saveFile,
        getFiles,
        // copyFileClipboard,
    ];
    commands.forEach(command => command());
};
exports.default = registerFileSystemCommands;
//# sourceMappingURL=fileSystemCommands.js.map