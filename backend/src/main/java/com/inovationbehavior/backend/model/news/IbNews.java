package com.inovationbehavior.backend.model.news;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Date;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class IbNews {
    private Integer id;
    private String titleZn;
    private String titleEn;
    private String cover;
    private String urlZn;
    private String urlEn;
    private Date time;
}
