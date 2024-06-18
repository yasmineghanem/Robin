import * as vscode from 'vscode';


// go to line
const goToLine = () => vscode.commands.registerCommand('robin.goToLine', (data) => {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
        const position = new vscode.Position(data.line, data.character ?? 0);
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

        //   vscode.window.showErrorMessage('No active text editor.');
    }
});


// register commands
const registerIDECommands = () => {
    const commands = [
        goToLine
    ];

    commands.forEach(command => command());
};

export default registerIDECommands;