package com.inovationbehavior.backend.controller;

import com.inovationbehavior.backend.exception.BannedWordException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.util.Map;

/**
 * 全局异常处理：违禁词等业务异常统一返回 400 + JSON。
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(BannedWordException.class)
    public ResponseEntity<Map<String, String>> handleBannedWord(BannedWordException ex) {
        return ResponseEntity
                .status(HttpStatus.BAD_REQUEST)
                .body(Map.of("message", ex.getMessage()));
    }
}
