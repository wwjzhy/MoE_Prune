# Algorithm references

REAM grouping, neuron signatures and joint activation/weight alignment are
adapted from https://github.com/SamsungSAILMontreal/ream at
84a3030716a0059589e9d10e2ea049e32b76cfa6.

MIT License

Copyright (c) 2026. Samsung Electronics Co., Ltd. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

HC-SMoE's output-mean average-linkage/frequency baseline is independently
implemented from https://github.com/wazenmai/HC-SMoE at
bd77eba1b55ca58606a47e2d0807a24218da73c3 and its scripts/README.md.

W4 offset/nibble layout follows compressed-tensors' documented signed pack
format; the implementation here is independent. No code from that project
is vendored. LiveCodeBench and EvalPlus are separate pinned dependencies
and retain their own licenses.
