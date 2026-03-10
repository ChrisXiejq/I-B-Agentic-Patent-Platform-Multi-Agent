package com.inovationbehavior.backend.exception;

/**
 * 用户输入包含违禁词时抛出，由全局异常处理返回 400。
 */
public class BannedWordException extends RuntimeException {

    private static final String DEFAULT_MESSAGE = "您的输入包含不当内容，请修改后重试。";

    public BannedWordException() {
        super(DEFAULT_MESSAGE);
    }

    public BannedWordException(String message) {
        super(message != null ? message : DEFAULT_MESSAGE);
    }

    public BannedWordException(String message, String matchedWord) {
        super(message != null ? message : DEFAULT_MESSAGE + " (matched: " + matchedWord + ")");
    }
}
