package com.inovationbehavior.backend.model.intelligence;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class Lecturer {
    Long _id;
    String name_en;
    String name_zn;
    String name_de;
    Long course_id;
}
