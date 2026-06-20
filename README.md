# TensorBuster: Where MCP and C2 frameworks collide
The days of getting past security solutions using Cobalt Strike and nothing else are over. In order to get past real-time monitoring, one needs to design a C2 framework that's not only polymorphic but also, critically, nondeterministic. Legacy polymorphism like that of Sliver may get you past traditional AV solutions, but the only way to get past modern EDRs like CrowdStrike Falcon or Palo Alto XDR on a consistent basis is to get creative, especially with the implementation of AI by the EDR solutions themselves.

While going through the COAE course, the combination of tensor steganography in the `AI Data Attacks` module and MCP in the `Attacking AI: Application and System` module gave me an idea: What if it's possible to use MCP as a C2 connection, LLMs as the implants, and tensor steganography as a C2 stager?

TensorBuster is the result of this experiment. It's an atttempt to build a complete C2 framework specifically for the age of AI, where tensor steganography is used to hide one LLM inside another and/or to hide an LLM inside image tensors.

## WARNING: UNTESTED
Because I have yet to be offered a real-world engagement that would allow proper testing of this beast and isn't HTB-confidential stuff, I currently do not have the system resources to adequately test this. As such, those with access to more than one powerful machine capable of running powerful LLMs is going to need to test this on either a production network, a large room with many physical machines in it, or an AI lab environment that's significant enough to be .
PRs and issue reports are welcome! If during testing any issues are discovered, feel free to report them and I'll look into it.

## What's implemented
* MCP core
* Pivoting (via the `mcp_pivot` tool, which LLMs can use to spin up clones of the original MCP server to cross subnet boundaries)
* Tensor steganography (defined via the `encode_lsb`, `decode_lsb`, `encode_lsb_from_image`, `payload_enc`, `export_encoded`, and `import_image` MCP tools)
* Exfiltration (via the `load_file` tool)
* Command execution (via the `run_system_command` tool)

## What still needs work
* Beaconing / sleep obfuscation
* On-the-fly payload generation/compilation
* GUI