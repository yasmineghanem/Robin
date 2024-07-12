"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const cors_1 = __importDefault(require("cors"));
const express_1 = __importDefault(require("express"));
const codeRouter_1 = __importDefault(require("./codeRouter"));
const fileSystemRouter_1 = __importDefault(require("./fileSystemRouter"));
const gitRouter_1 = __importDefault(require("./gitRouter"));
const IDERouter_1 = __importDefault(require("./IDERouter"));
const server = (0, express_1.default)();
// middleware
server.use((0, cors_1.default)());
server.use(express_1.default.json());
server.use('/file-system', fileSystemRouter_1.default);
server.use('/ide', IDERouter_1.default);
server.use('/code', codeRouter_1.default);
server.use('/git', gitRouter_1.default);
// export the server
exports.default = server;
//# sourceMappingURL=server.js.map