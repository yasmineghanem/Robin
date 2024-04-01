// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';
const axios = require('axios');
const http = require("http");

const express = require("express");
const cors = require("cors");

function get_endpoint(editor: vscode.TextEditor): vscode.Position {
	const lastLine = editor.document.lineAt(editor.document.lineCount - 1);
	const endPosition = new vscode.Position(editor.document.lineCount - 1, lastLine.range.end.character);
	return endPosition;
}
function get_currentpoint(editor: vscode.TextEditor): vscode.Position {
	const position = editor.selection.active;
	return position;
}
// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed
export function activate(context: vscode.ExtensionContext) {
	let declareVariable = vscode.commands.registerCommand('robin.declareVariable', (args, body) => {
		const editor = vscode.window.activeTextEditor;
		// {
		// 	"operation": null,
		// 	"parameters": [
		// 		{
		// 			"name": "x",
		// 			"type": null,
		// 			"default": 10
		// 		}
		// 	]
		// }
		const parameters = body.parameters;
		let name = parameters[0]['name'];
		let value = parameters[0]['default'];
		// Check if an editor is open
		if (editor) {
			const line = `\n${name} = ${JSON.stringify(value)}`;

			editor.edit(editBuilder => {
				editBuilder.insert(get_currentpoint(editor), line);
			});
		} else {
			vscode.window.showErrorMessage('No active text editor.');
		}

	});

	let declareFunction = vscode.commands.registerCommand('robin.declareFunction', (args, body) => {
		const editor = vscode.window.activeTextEditor;
		// {
		// 	"operation": null,
		// 	"parameters": [
		// 		null, // first paremeter is the return type
		// 		"add", // second parameter is the function name
		// 		{
		// 			"name": "x",
		// 			"type": null,
		// 			"default": 0
		// 		},
		// 		{
		// 			"name": "y",
		// 			"type": null,
		// 			"default": 10
		// 		}
		// 	]
		// }
		const parameters = body.parameters;
		let type = parameters[0];
		let name = parameters[1];
		let line1 = "def " + name + "(";
		for (let i = 2; i < parameters.length; i++) {
			if (i - 2) line1 += ",";
			line1 += parameters[i]['name'] + ' = ' + JSON.stringify(parameters[i]['default']);
		}
		line1 += "):";
		// Check if an editor is open
		if (editor) {
			editor.edit(editBuilder => {
				editBuilder.insert(get_currentpoint(editor), `\n${line1}`);
				editBuilder.insert(get_currentpoint(editor), '\n	return None');
			});
		} else {
			vscode.window.showErrorMessage('No active text editor.');
		}
	});

	let declareClass = vscode.commands.registerCommand('robin.declareClass', (args, body) => {
		const editor = vscode.window.activeTextEditor;
		// {
		// 	"operation": null,
		// 	"parameters": [
		// 		"person" // first parameter is the class name
		// 	]
		// }

		let parameters = body.parameters;
		let name = parameters[0];
		let line = `\nclass ${name}:`;
		// Check if an editor is open
		if (editor) {
			editor.edit(editBuilder => {
				editBuilder.insert(get_currentpoint(editor), line);
			});
		} else {
			vscode.window.showErrorMessage('No active text editor.');
		}

	});

	let goToLocation = vscode.commands.registerCommand('robin.goToLocation', (data) => {
		const editor = vscode.window.activeTextEditor;
		if (editor) {
			const position = new vscode.Position(data.line, data.character ?? 0);
			editor.selection = new vscode.Selection(position, position);
			editor.revealRange(new vscode.Range(position, position));
		} else {
			vscode.window.showErrorMessage('No active text editor.');
		}
	});

	const server = express();
	server.use(cors());
	server.use(express.json());

	server.post("/declare-var", (req: any, res: any) => {
		const data = req.body;
		const args = req.query;
		vscode.commands.executeCommand("robin.declareVariable", args, data).then(
			() => {
				res.writeHead(200, { "Content-Type": "text/plain" });
				res.end("Command executed successfully");
			},
			(err) => {
				res.writeHead(500, { "Content-Type": "text/plain" });
				res.end("Failed to execute command");
			}
		);

	});
	server.post("/declare-func", (req: any, res: any) => {
		const data = req.body;
		const args = req.query;
		vscode.commands.executeCommand("robin.declareFunction", args, data).then(
			() => {
				res.writeHead(200, { "Content-Type": "text/plain" });
				res.end("Command executed successfully");
			},
			(err) => {
				res.writeHead(500, { "Content-Type": "text/plain" });
				res.end("Failed to execute command");
			}
		);
	});
	server.post("/declare-class", (req: any, res: any) => {
		const data = req.body;
		const args = req.query;
		vscode.commands.executeCommand("robin.declareClass", args, data).then(
			() => {
				res.writeHead(200, { "Content-Type": "text/plain" });
				res.end("Command executed successfully");
			},
			(err) => {
				res.writeHead(500, { "Content-Type": "text/plain" });
				res.end("Failed to execute command");
			}
		);
	});

	server.get("/file-name", (req: any, res: any) => {
		const editor = vscode.window.activeTextEditor;
		if (editor) {
			const fileName = editor.document.fileName;
			// get file name not whole path
			const fileNameArr = fileName.split("\\");

			res.writeHead(200, { "Content-Type": "text/plain" });
			res.end(fileNameArr[fileNameArr.length - 1]);
		} else {
			res.writeHead(500, { "Content-Type": "text/plain" });
			res.end("No active text editor");
		}
	});

	server.post("/go-to-location", (req: any, res: any) => {
		const data = req.body;
		vscode.commands.executeCommand("robin.goToLocation", data).then(
			() => {
				res.writeHead(200, { "Content-Type": "text/plain" });
				res.end("Command executed successfully");
			},
			(err) => {
				res.writeHead(500, { "Content-Type": "text/plain" });
				res.end("Failed to execute command");
			}
		);
	});
	server.listen(3000, () => {
		console.log("Server listening on port 3000");
	});
}
// This method is called when your extension is deactivated
export function deactivate() { }
