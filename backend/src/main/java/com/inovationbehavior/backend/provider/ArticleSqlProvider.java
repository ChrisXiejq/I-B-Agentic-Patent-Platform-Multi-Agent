package com.inovationbehavior.backend.provider;

import com.inovationbehavior.backend.model.economic.Article;
import org.apache.ibatis.jdbc.SQL;

public class ArticleSqlProvider {

    public String updateArticleSql(final Article article) {
        return new SQL() {{
            UPDATE("article");

            if (article.getType() != null) {
                SET("type = #{article.type}");
            }
            if (article.getTitle_en() != null) {
                SET("title_en = #{article.title_en}");
            }
            if (article.getTitle_zn() != null) {
                SET("title_zn = #{article.title_zn}");
            }
            if (article.getAbstract_en() != null) {
                SET("abstract_en = #{article.abstract_en}");
            }
            if (article.getAbstract_zn() != null) {
                SET("abstract_zn = #{article.abstract_zn}");
            }
            if (article.getAbstract_de() != null) {
                SET("abstract_de = #{article.abstract_de}");
            }
            if (article.getUrl() != null) {
                SET("url = #{article.url}");
            }

            WHERE("_id = #{article._id}");
        }}.toString();
    }
}

