package com.inovationbehavior.backend.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.util.ArrayList;
import java.util.List;

/**
 * app.ai 配置项：违禁词列表及拒绝提示文案。
 */
@ConfigurationProperties(prefix = "app.ai")
public class AppAiProperties {

    /** 违禁词列表，命中任一则拒绝请求（大小写不敏感） */
    private List<String> bannedWords = new ArrayList<>();

    /** 命中违禁词时返回的提示文案，为空则使用默认文案 */
    private String bannedWordsRejectMessage;

    public List<String> getBannedWords() {
        return bannedWords;
    }

    public void setBannedWords(List<String> bannedWords) {
        this.bannedWords = bannedWords != null ? bannedWords : new ArrayList<>();
    }

    public String getBannedWordsRejectMessage() {
        return bannedWordsRejectMessage;
    }

    public void setBannedWordsRejectMessage(String bannedWordsRejectMessage) {
        this.bannedWordsRejectMessage = bannedWordsRejectMessage;
    }
}
