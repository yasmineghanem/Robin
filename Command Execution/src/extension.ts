// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from "vscode";
import server from "./communication/server";
import registerAllCommands from "./commands/commands";
import path from 'path';
require('dotenv').config({
	path: path.resolve(__dirname, '../.env')
});  // Load environment variables from .env


// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed
export function activate(context: vscode.ExtensionContext) {
	// register all commands
	registerAllCommands();
	server.listen(process.env.SERVER_HOST_PORT || 3000, () => {
		// activate extension by executing hello world command
		vscode.commands.executeCommand("robin.activate");
		console.log("Server listening on port " + process.env.SERVER_HOST_PORT || 3000);

		// on any request log
		server.use((req, res, next) => {
			console.log(req.method, req.url);
			next();
		});
	});
}

// This method is called when your extension is deactivated
export function deactivate() { }
