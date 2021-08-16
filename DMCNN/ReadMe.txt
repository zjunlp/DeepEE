1. train_XXX : Train function.
2. tensor_XXX: Convert data to input tensor.
3. tagger_XXX: Tagger function.
4. DMCNN_XXX: Graph model funtion.
5. xxx_preinfo: Test/sort/plot

评测时候先将中间结果写入文件，再进行评测相关的处理.
训练过程中输出的PRF值不是最终的PRF，最终的是在排序之后，分不同TOP输出的PRF.
这些文件是学生复现模型时候写的原始文件，有一些不太规范的地方请见谅.