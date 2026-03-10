package com.inovationbehavior.backend.ai.app;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class IBAppTest {

    @Autowired
    private IBApp ibApp;

    @Test
    void doChatWithRag() {
        String ans = ibApp.doChatWithRag("What share of European inventions come from universities?", "test-rag-1");
        assertNotNull(ans);
    }
}