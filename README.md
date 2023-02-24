# pfg-webui
[PFG](https://github.com/laksjdjf/pfg)(Prompt Free Generation) is a method of guiding with an image by concatenating the image into a text encoder hidden states.
It can generate a variety of images without prompting.

# Method
1. Convert image into (786,) vector by [wd14tagger](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2) (output of last pooling layer)
2. Convert the vector to (num_tokens, 768 or 1024) tensor by pretrained linear.
3. Concatenate text encoder hidden states and it.

# Usage
+ Install this extension in the same way as other extensions.
+ Download [wd14tagger](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2).
+ Download [pretrained model](https://huggingface.co/furusu/PFG) and put it in ./models.
+ You can see menu bar of "PFG" on txt2img tab.

# Example
![image](example.png)
