const { GoogleGenerativeAI } = require('@google/generative-ai');
const { GoogleGenerativeAIEmbeddings } = require('@langchain/google-genai');
const { TaskType } = require('@google/generative-ai');
const { MemoryVectorStore } = require('langchain/vectorstores/memory');

class GeminiGenerate {
  constructor(apiKey) {
    this.genAI = new GoogleGenerativeAI(apiKey);

    this.embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey,
      model: 'text-embedding-004', // 768 dimensions
      taskType: TaskType.RETRIEVAL_DOCUMENT,
      title: 'Document title',
    });
  }

  async filter_functions(functions, instruction) {
    const documents = functions.map((f) => {
      const { parameters, ...fDescription } = f.function;
      return JSON.stringify(fDescription);
    });

    const vectorStore = await MemoryVectorStore.fromTexts(documents, [], this.embeddings);

    // 128 (max no of functions accepted by openAI function calling)
    const retrievedDocuments = await vectorStore.similaritySearch(instruction, 10);
    var selectedFunctions = [];
    retrievedDocuments.forEach((document) => {
      const pDocument = JSON.parse(document.pageContent);
      const findF = functions.find(
        (f) => f.function.name === pDocument.name && f.function.description === pDocument.description,
      );
      if (findF) {
        selectedFunctions = selectedFunctions.concat(findF);
      }
    });

    return selectedFunctions;
  }

  async process_user_instruction(functions, instruction) {
    //console.log(functions.map((f) => f.function.name));
    // Define the function call format
    const fn = `{"name": "function_name"}`;

    // Prepare the function string for the system prompt
    const fnStr = functions.map((f) => JSON.stringify(f)).join('\n');

    // Define the system prompt
    const systemPrompt = `
        You are a helpful assistant with access to the following functions:

        ${fnStr}

        To use these functions respond with, only output function names, ignore arguments needed by those functions:

        <multiplefunctions>
            <functioncall> ${fn} </functioncall>
            <functioncall> ${fn} </functioncall>
            ...
        </multiplefunctions>

        Edge cases you must handle:
        - If there are multiple functions that can fullfill user request, list them all.
        - If there are no functions that match the user request, you will respond politely that you cannot help.
        - If the user has not provided all information to execute the function call, choose the best possible set of values. Only, respond with the information requested and nothing else.
        - If asked something that cannot be determined with the user's request details, respond that it is not possible to fulfill the request and explain why.
    `;

    const model = this.genAI.getGenerativeModel({
      model: 'gemini-1.5-pro-latest',
      systemInstruction: {
        role: 'system',
        parts: [{ text: systemPrompt }],
      },
    });

    // Prepare the messages for the language model

    const request = {
      contents: [{ role: 'user', parts: [{ text: instruction }] }],
    };

    // Invoke the language model and get the completion
    const completion = await model.generateContent(request);

    const content = completion.response.candidates[0].content.parts[0].text.trim();

    // Extract function calls from the completion
    const extractedFunctions = this.extractFunctionCalls(content);

    return extractedFunctions;
  }

  extractFunctionCalls(completion) {
    let content = typeof completion === 'string' ? completion : completion.content;

    // Multiple functions lookup
    const mfnPattern = /<multiplefunctions>(.*?)<\/multiplefunctions>/s;
    const mfnMatch = content.match(mfnPattern);

    // Single function lookup
    const singlePattern = /<functioncall>(.*?)<\/functioncall>/s;
    const singleMatch = content.match(singlePattern);

    let functions = [];

    if (!mfnMatch && !singleMatch) {
      // No function calls found
      return null;
    } else if (mfnMatch) {
      // Multiple function calls found
      const multiplefn = mfnMatch[1];
      const fnMatches = [...multiplefn.matchAll(/<functioncall>(.*?)<\/functioncall>/gs)];
      for (let fnMatch of fnMatches) {
        const fnText = fnMatch[1].replace(/\\/g, '');
        try {
          functions.push(JSON.parse(fnText));
        } catch {
          // Ignore invalid JSON
        }
      }
    } else {
      // Single function call found
      const fnText = singleMatch[1].replace(/\\/g, '');
      try {
        functions.push(JSON.parse(fnText));
      } catch {
        // Ignore invalid JSON
      }
    }
    return functions;
  }
}

module.exports = GeminiGenerate;
