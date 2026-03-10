package com.inovationbehavior.backend.model.economic;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * 通用文章类
 * 用于存储仅需要标题、作者、概要和外链的文章信息。
 *
 * @author litianlololo
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class Article {
    Long _id;
    String title_en;
    String title_zn;
    String title_de;
    String abstract_en;
    String abstract_zn;
    String abstract_de;
    String url;
    String type;
    List<Author> authors;

    public Article(String titleEn, String titleZn, String titleDe, String abstractEn, String abstractZn, String abstractDe, String url, String type) {
        this.title_en = titleEn;
        this.title_zn = titleZn;
        this.title_de = titleDe;
        this.abstract_en = abstractEn;
        this.abstract_zn = abstractZn;
        this.abstract_de = abstractDe;
        this.url = url;
        this.type = type;
    }
}
