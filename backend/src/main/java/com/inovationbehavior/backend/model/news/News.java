package com.inovationbehavior.backend.model.news;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.sql.Date;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class News {
    private Integer id;
    private String titleZn;
    private String titleEn;
    private String titleDe;
    private String containsZn;
    private String containsEn;
    private String containsDe;
    private Date time;
}
