#!/usr/bin/env python 3.6.8
# -*- coding: utf-8 -*-
# @Time    : 2021/5/24 18:55
# @Author  : fangwenchu
# @File    : stkgviews.py

from django.shortcuts import render
from .stkgqa import XiaoFangQA


def kbqa(request):
    return render(request, 'kbqa.html')


def getAnswer(request):
    request.encoding = "utf-8"
    question = request.GET["question"]
    if not question:
        question = " "
    qa = XiaoFangQA.XiaoFangQA()
    print("question:", question)
    answer = qa.chat_main(question)
    return render(request, "show-answer.html", {"answer": answer})
