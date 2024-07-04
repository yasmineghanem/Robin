import * as vscode from 'vscode';
import registerFileSystemCommands from './fileSystemCommands';
import registerIDECommands from './IDECommands';
import registerCodeCommands from './codeCommands';

// register commands
// function get_endpoint(editor: vscode.TextEditor): vscode.Position {
//   const lastLine = editor.document.lineAt(editor.document.lineCount - 1);
//   const endPosition = new vscode.Position(editor.document.lineCount - 1, lastLine.range.end.character);
//   return endPosition;
// }
function get_currentpoint(editor: vscode.TextEditor): vscode.Position {
  const position = editor.selection.active;
  return position;
}

const activateRobin = () => vscode.commands.registerCommand('robin.activate', () => {
  vscode.window.showInformationMessage('Robin Activated!');
});

// const declareVariable = () => vscode.commands.registerCommand('robin.declareVariable', (args, body) => {
//   const editor = vscode.window.activeTextEditor;
//   // {
//   // 	"operation": null,
//   // 	"parameters": [
//   // 		{
//   // 			"name": "x",
//   // 			"type": null,
//   // 			"default": 10
//   // 		}
//   // 	]
//   // }
//   const parameters = body.parameters;
//   let name = parameters[0]['name'];
//   let value = parameters[0]['default'];
//   // Check if an editor is open
//   if (editor) {
//     const line = `\n${name} = ${JSON.stringify(value)}`;

//     editor.edit(editBuilder => {
//       editBuilder.insert(get_currentpoint(editor), line);
//     });
//   } else {
//     vscode.window.showErrorMessage('No active text editor.');
//   }

// });

// const declareFunction = () => vscode.commands.registerCommand('robin.declareFunction', (args, body) => {
//   const editor = vscode.window.activeTextEditor;
//   // {
//   // 	"operation": null,
//   // 	"parameters": [
//   // 		null, // first paremeter is the return type
//   // 		"add", // second parameter is the function name
//   // 		{
//   // 			"name": "x",
//   // 			"type": null,
//   // 			"default": 0
//   // 		},
//   // 		{
//   // 			"name": "y",
//   // 			"type": null,
//   // 			"default": 10
//   // 		}
//   // 	]
//   // }
//   const parameters = body.parameters;
//   let type = parameters[0];
//   let name = parameters[1];
//   let line1 = "def " + name + "(";
//   for (let i = 2; i < parameters.length; i++) {
//     if (i - 2) line1 += ",";
//     line1 += parameters[i]['name'] + ' = ' + JSON.stringify(parameters[i]['default']);
//   }
//   line1 += "):";
//   // Check if an editor is open
//   if (editor) {
//     editor.edit(editBuilder => {
//       editBuilder.insert(get_currentpoint(editor), `\n${line1}`);
//       editBuilder.insert(get_currentpoint(editor), '\n	return None');
//     });
//   } else {
//     vscode.window.showErrorMessage('No active text editor.');
//   }
// });

// const declareClass = () => vscode.commands.registerCommand('robin.declareClass', (args, body) => {
//   const editor = vscode.window.activeTextEditor;

//   let parameters = body.parameters;
//   let name = parameters[0];
//   let line = `\nclass ${name}:`;
//   // Check if an editor is open
//   if (editor) {
//     editor.edit(editBuilder => {
//       editBuilder.insert(get_currentpoint(editor), line);
//     });
//   } else {
//     vscode.window.showErrorMessage('No active text editor.');
//   }

// });

const goToLocation = () => vscode.commands.registerCommand('robin.goToLocation', (data) => {
  const editor = vscode.window.activeTextEditor;
  if (editor) {
    const position = new vscode.Position(data.line, data.character ?? 0);
    editor.selection = new vscode.Selection(position, position);
    editor.revealRange(new vscode.Range(position, position));
  } else {
    vscode.window.showErrorMessage('No active text editor.');
  }
});


const fileName = () => vscode.commands.registerCommand('robin.fileName', () => {
  const editor = vscode.window.activeTextEditor;
  if (editor) {
    const fileName = editor.document.fileName;
    const fileNameArr = fileName.split("\\");
    return (fileNameArr[fileNameArr.length - 1]);
  } else {
    vscode.window.showErrorMessage('No active text editor.');
  }
});


// register commands 
const registerAllCommands = () => {
  const commands = [
    activateRobin,
    // declareClass,
    goToLocation,
    fileName
  ];

  commands.forEach(command => command());

  registerFileSystemCommands();
  registerIDECommands();
  registerCodeCommands();
};
export default registerAllCommands;