// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';
const axios = require('axios');

function get_endpoint( editor :vscode.TextEditor):vscode.Position {
	const lastLine = editor.document.lineAt(editor.document.lineCount - 1);
	const endPosition = new vscode.Position(editor.document.lineCount - 1, lastLine.range.end.character);
	return endPosition;
}
// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed
export function activate(context: vscode.ExtensionContext) {

	// Use the console to output diagnostic information (console.log) and errors (console.error)
	// This line of code will only be executed once when your extension is activated
	console.log('Congratulations, your extension "robin" is now active!');

	// The command has been defined in the package.json file
	// Now provide the implementation of the command with registerCommand
	// The commandId parameter must match the command field in package.json
	let disposable = vscode.commands.registerCommand('robin.helloWorld', () => {
		// The code you place here will be executed every time your command is executed
		// Display a message box to the user
		// vscode.window.showInformationMessage('Hello World from Robin!');
		// vscode.commands.executeCommand('editor.action.addCommentLine');
		const editor = vscode.window.activeTextEditor;

        // Check if an editor is open
        if (editor) {
            // Insert code line y = 400 at the end of the file
            editor.edit(editBuilder => {
                editBuilder.insert(get_endpoint(editor), '\ny = 400');
				editBuilder.insert(get_endpoint(editor), '\nprint(y)');
				editBuilder.insert(get_endpoint(editor),  `\ndef add(x, y):`);
				editBuilder.insert(get_endpoint(editor), '\n	return x + y');
				editBuilder.insert(get_endpoint(editor), '\nprint(add(2, 3))');
            });
			
			vscode.window.showInformationMessage('line added succufully.');
        } else {
            vscode.window.showErrorMessage('No active text editor.');
        }

	});
	
	context.subscriptions.push(disposable);
}

// This method is called when your extension is deactivated
export function deactivate() {}
