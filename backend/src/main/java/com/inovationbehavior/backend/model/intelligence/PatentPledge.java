package com.inovationbehavior.backend.model.intelligence;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class PatentPledge {
    Long _id;
    String title_en;
    String title_zn;
    String title_de;
    String url_en;
    String url_zn;
    String url_de;
    List<PPAuthor> lecturers;
}
