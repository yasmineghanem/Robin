import * as vscode from 'vscode';
import fs from "fs";
import { COPY_FILE, CREATE_DIRECTORY, CREATE_FILE, DELETE_FILE, RENAME, SAVE } from '../constants/fileSystem';



/**
 * 
 * defining some utilities
 */

// check if the file or directory exists
// if check with and without extension
const fileExists = (path: string) => {
    if (fs.existsSync(path)) {
        return true;
    }
    const files = fs.readdirSync(vscode.workspace.rootPath?.toString() || '');
    const file = files.find(file => file.split('.')[0] === path);
    if (file) {
        return true;
    }
    return false;
};







// create new file
// sample input => 
// {
//     "fileName": "new file",
//     "extension": "/py",
//     "content": "file content"
// }
const createFile = () => vscode.commands.registerCommand(CREATE_FILE, (args) => {
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
const createDirectory = () => vscode.commands.registerCommand(CREATE_DIRECTORY, (args) => {
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
    vscode.commands.registerCommand(COPY_FILE, (args) => {
        const source = args.source;
        const destination = args.destination;
        const sourcePath = `${vscode.workspace.rootPath}\\${source}`;
        const destinationPath = `${vscode.workspace.rootPath}\\${destination}`;
        let path = "";

        // check if the source file doesn't have an extension, get the first the file the matches the source
        if (source.split('.').pop() === source) {

            const files = fs.readdirSync(vscode.workspace.rootPath?.toString() || '');
            const file = files.find(file => file.split('.')[0] === source);
            if (file) {
                const content = fs.readFileSync(`${vscode.workspace.rootPath}\\${file}`, 'utf-8');
                if (destination) {
                    path = destinationPath;
                    fs.writeFileSync(destinationPath, content);
                } else {
                    // add copy before extenstion
                    path = `${vscode.workspace.rootPath}\\${source}-copy.${file.split('.').pop()}`;
                    fs.writeFileSync(path, content);
                }

                return {
                    success: true,
                    path
                };
            }
        }
        else {

            if (fs.existsSync(sourcePath)) {
                const content = fs.readFileSync(sourcePath, 'utf-8');
                if (destination) {
                    fs.writeFileSync(destinationPath, content);
                } else {
                    // add copy before extension
                    path = `${vscode.workspace.rootPath}\\${source.split('.')[0]}-copy.${source.split('.').pop()}`;
                    fs.writeFileSync(path, content);
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
    vscode.commands.registerCommand(CREATE_DIRECTORY, (args) => {
        const source = args.source;
        const destination = args.destination;
        const overwrite = args.overwrite ?? false;

        const sourcePath = `${vscode.workspace.rootPath}\\${source}`;

        if (fs.existsSync(sourcePath
        )) {
            if (destination) {
                vscode.workspace.fs.copy(
                    vscode.Uri.file(sourcePath),
                    vscode.Uri.file(`${vscode.workspace.rootPath}\\${destination}`)
                    ,

                );
            }
            else {
                vscode.workspace.fs.copy(
                    vscode.Uri.file(sourcePath),
                    vscode.Uri.file(`${vscode.workspace.rootPath}\\${source} copy`)
                );
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
    vscode.commands.registerCommand(DELETE_FILE, (args) => {
        const source = args.source;
        const sourcePath = `${vscode.workspace.rootPath}\\${source}`;
        if (fileExists(sourcePath
        )) {
            vscode.workspace.fs.delete(
                vscode.Uri.file(sourcePath),
                { recursive: true }
            );
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
    vscode.commands.registerCommand(RENAME, (args) => {
        const source = args.source;
        const destination = args.destination;
        // const overwrite = args.overwrite ?? false;
        const sourcePath = `${vscode.workspace.rootPath}\\${source}`;
        const destinationPath = `${vscode.workspace.rootPath}\\${destination}`;

        if (fileExists(sourcePath)) {
            vscode.workspace.fs.rename(
                vscode.Uri.file(sourcePath),
                vscode.Uri.file(destinationPath),
                // { overwrite }
            );
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
    vscode.commands.registerCommand(SAVE, (args) => {
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
            };
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
        // copyFileClipboard,
    ];

    commands.forEach(command => command());
};


export default registerFileSystemCommands;
