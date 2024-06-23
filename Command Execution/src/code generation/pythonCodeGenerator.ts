import { CodeGenerator } from "./codeGenerator";

export class PythonCodeGenerator extends CodeGenerator {
    declareVariable(name: string, initialValue?: any): string {
        return `${name} = ${initialValue !== undefined ? initialValue : 'None'}`;
    }

    declareFunction(
        name: string,
        returnType: string,
        parameters: { name: string, type: string }[],
        body: string[]
    ): string {
        const params = parameters.map(p => p.name).join(', ');
        const bodyCode = this.wrapInCodeBlock(body);
        return `def ${name}(${params}) -> ${returnType}:\n${bodyCode}`;
    }

    declareClass(
        name: string,
        properties: { name: string, type: string }[],
        methods: string[]
    ): string {
        const props = properties.map(p => `self.${p.name} = None`).join('\n        ');
        const initMethod = `def __init__(self):\n        ${props}`;
        const methodCode = methods.join('\n\n    ');
        return `class ${name}:\n    ${initMethod}\n\n    ${methodCode}`;
    }

    generateImport(module: string, entities: string[]): string {
        return `from ${module} import ${entities.join(', ')}`;
    }

    wrapInCodeBlock(lines: string[]): string {
        return lines.map(line => `    ${line}`).join('\n');
    }
}
