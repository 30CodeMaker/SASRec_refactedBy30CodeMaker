​	这是由30CodeMaker基于论文《Self-Attentive Sequential Recommendation》并且基于pmixer作者的pytorch版本的SASRec代码再次重构的，因为30CodeMaker在阅读pmixer作者的代码时觉得有些地方写的对读者挺不友好（仅仅30CodeMaker个人观点），因此萌生了对pmixer作者的代码重构的想法。

​	30CodeMaker在重构过程中个人觉得了pmixer版本的代码没有严格按照原论文的神经网络结构进行搭建，使用了少许trick，因此30CodeMaker版本的代码是严格根据原论文的神经网络结构进行搭建的（仅仅根据30CodeMaker对原论文的理解），最后出来的结果显示比pmixer版本的效果稍微差一点点（可能充分模仿pmixer中代码运用的tricks即可与其持平，当然30CodeMaker没有去尝试，感兴趣的读者可以一试，如果可以的话可以反馈给30CodeMaker，谢谢！）

​	欢迎各位读者指正代码中的错误，可以在github评论中联系我或者联系邮箱2523733157@qq.com

。