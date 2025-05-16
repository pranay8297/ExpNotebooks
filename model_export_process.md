# DBNet model export process from PaddleOCR

1. Train the model using paddlepaddle-gpu 2.6.2 version  
2. Export the inference model using infer tools provided in paddleOCR library with right config

Step 1: convert the model from paddle to onnx

1. Uninstall paddlepaddle-gpu 2.6.2 gpu  
2. Install paddlepaddle 3.0.0.b (The reason we have to do this, for cuda 12.4 there is no paddlepaddle 3.x gpu library, hence we have to train on 2.6.2 and export using 3.0.0.b as the onnx converter library is only compatible with 3.0.0.b)  
3. \`\`\`  
   \!pip install paddle2onnx  
   \!pip install onnx  
   \!pip install onnxruntime  
   \`\`\`  
4. \!paddle2onnx \\  
     \--model\_dir ./inference/dbnet\_r18/ \\  
     \--model\_filename inference.pdmodel \\  
     \--params\_filename inference.pdiparams \\  
     \--save\_file /content/dbnet\_test.onnx \\  
     \--opset\_version 14

5. Test with the following snippet and verify the outputs:   
   \`\`\`  
   import onnxruntime as ort  
   import numpy as np  
     
   import onnx  
     
   model \= onnx.load("/content/dbnet\_test.onnx")  
   onnx.checker.check\_model(model)  \# raises if invalid  
 


   \# Load session

   sess \= ort.InferenceSession("/content/dbnet\_test.onnx")

   

   \# Prepare dummy input matching your model’s input shape

   input\_name \= sess.get\_inputs()\[0\].name

   dummy \= np.random.randn(1,3,640,640).astype(np.float32)

   

   \# Run inference

   outputs \= sess.run(None, {input\_name: dummy})

   print(outputs\[0\].shape)  \# e.g., bounding box logits  
   \`\`\`

Step 2: Convert the model from onnx to tf and tflite

1. \`\`\`  
   \!pip install onnx2tf  
   \!pip install onnx\_graphsurgeon  
   \!pip install ai-edge-litert  
   \!pip install \-U sng4onnx\>=1.0.4  
   \`\`\`  
2. \`\`\`  
   \# fix the input shape to 640, 640  
   \!python \-m onnxruntime.tools.make\_dynamic\_shape\_fixed \\  
     \--input\_name x \\  
     \--input\_shape 1,3,640,640 \\  
     /content/dbnet\_test.onnx /content/dbnet\_test\_fixed.onnx  
   \`\`\`  
3. \`\`\`  
   \# convert the fixed model to tf  
   \!onnx2tf \-i /content/dbnet\_test\_fixed.onnx \-o /content/tf\_dbnet  
   \`\`\`

4. Verify tf\_dbnet directory is created or not  
5. Refer to this notebooks sections for more detailed code  
   https://colab.research.google.com/drive/1qksAVy5ZxJADXjAo0\_\_fNo9qoPY-WMVl?usp=sharing

# Parseq to torchscript

1. \`\`\`  
   \!git clone https://github.com/baudm/parseq.git  
   \# \!pip install \-r parseq/requirements/core.txt  
   \!pip install pytorch-lightning==2.2.0.post0  
   \!pip install lightning-utilities==0.10.1  
   \`\`\`  
2. \!pip install \-e ./parseq/  
3. Please edit the parseq/strhub/models/parseq/model.py file according to this file:   
   [https://drive.google.com/file/d/1-3uZsncFfnPtnTFJO6ZZAXjDh2CBSnKu/view?usp=sharing](https://drive.google.com/file/d/1-3uZsncFfnPtnTFJO6ZZAXjDh2CBSnKu/view?usp=sharing)  
4. \`\`\`  
   import torch  
   from torch import nn  
   from parseq.strhub.models.parseq.model import PARSeq  
   \`\`\`  
5. \`\`\`  
   par\_seq\_model \= PARSeq(  
       num\_tokens \= 97,  
           max\_label\_length \= 25,  
           img\_size \= \[32, 128\],  
           patch\_size \= \[4, 8\],  
           embed\_dim \= 384,  
           enc\_num\_heads \= 6,  
           enc\_mlp\_ratio \= 4,  
           enc\_depth \= 12,  
           dec\_num\_heads \= 12,  
           dec\_mlp\_ratio \= 4,  
           dec\_depth \= 1,  
           decode\_ar \= False,  
           refine\_iters \= 0,  
           dropout \= 0.1,  
   )  
   \`\`\`  
6. Access the weights state dict using this link please and download: [https://drive.google.com/file/d/1RFr5fQP1E1PTfFzBDWU0l7f\_8zu6h66E/view?usp=sharing](https://drive.google.com/file/d/1RFr5fQP1E1PTfFzBDWU0l7f_8zu6h66E/view?usp=sharing)  
   \`\`\`  
   state\_dict \= torch.load("{path to downloaded weights using the above link}")  
   par\_seq\_model.load\_state\_dict(state\_dict, strict=True)  
   \`\`\`  
7. \`\`\`  
   from typing import NamedTuple  
   from torch import nn  
   import torch  
     
   class Tokenizer(NamedTuple):  
       pad\_id: int  
       bos\_id: int  
       eos\_id: int  
     
   class MyModel(nn.Module):  
       def \_\_init\_\_(self, base\_model: nn.Module):  
           super().\_\_init\_\_()  
           self.model \= base\_model  
           self.model.decode\_ar \= False  
           self.model.refine\_iters \= 0  
           self.max\_length \= 14  
           \# TorchScript‐friendly tokenizer  
           self.tokenizer \= Tokenizer(pad\_id=96, bos\_id=95, eos\_id=0)  
     
       def forward(self, x: torch.Tensor) \-\> torch.Tensor:  
           \# Now both inputs and attributes are TorchScript types  
           return self.model( x, self.max\_length)  
         
     
   parseq\_my\_model \= MyModel(par\_seq\_model).eval()  
   \`\`\`  
8. \`\`\`  
   scripted \= torch.jit.script(parseq\_my\_model)  
   scripted.save("parseq\_scripted.pt") \# or path of your choice  
   \`\`\`  
   

     View the implementation in [onnx\_to\_tflite.ipynb](https://colab.research.google.com/drive/1asQCtbYX2EVmEiNd5wMOma0gdEUNsgQK?usp=sharing) **Parseq to direct torchscript** section