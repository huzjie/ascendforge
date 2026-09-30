# FAQ

**Q: 需要昇腾硬件吗？** A: 不需要。mock/cpu 后端在任何机器可跑；ascend 后端为 stub + 性能模型。

**Q: 需要安装依赖吗？** A: 不需要，零第三方依赖。

**Q: 能对接真实推理服务吗？** A: 可以，`openai` 后端对接任意 OpenAI 兼容端点。

**Q: 如何用 MCP？** A: `python -m ascendforge.integrations`，按 JSON-RPC over stdio 协议接入任意 MCP 客户端。
