package com.inovationbehavior.backend.mapper;

import com.inovationbehavior.backend.model.economic.Article;
import com.inovationbehavior.backend.model.economic.Author;
import com.inovationbehavior.backend.model.intelligence.Course;
import com.inovationbehavior.backend.model.intelligence.Lecturer;
import com.inovationbehavior.backend.model.intelligence.PPAuthor;
import com.inovationbehavior.backend.model.intelligence.PatentPledge;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface EconomicMapper {
    @Select("""
    SELECT count(*) FROM article
    WHERE type = #{type}
    """)
    Long count(@Param("type") String type);
    @Select("""
        SELECT _id, title_en,title_zn,title_de, abstract_en, abstract_zn, abstract_de , url,type
        FROM article
        WHERE type = #{type}
        LIMIT #{start},#{size}
    """)
    List<Article> getArticleByPage(@Param("start") Integer start, @Param("size") Integer size, @Param("type") String type);

    @Select("""
        SELECT _id, name_en, name_zn, name_de, article_id
        FROM article_author
        WHERE article_id = #{article_id}
    """)
    List<Author> getAuthorById(@Param("article_id") Long article_id);

    @Select("""
    SELECT * FROM article
    WHERE _id = #{article_id}
""")
    Article getArticleById(@Param("article_id") Long article_id);
}
