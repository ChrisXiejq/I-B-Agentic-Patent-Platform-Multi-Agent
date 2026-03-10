package com.inovationbehavior.backend.service.intf;

import com.inovationbehavior.backend.mapper.PatentMapper;
import com.inovationbehavior.backend.model.Patent;
import com.inovationbehavior.backend.model.Result;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.List;
import java.util.Map;

@Service
public interface PatentService {
    Patent getPatentByNo(String No);

    List<Patent> getPatentByKey(String company);

    List<String> getCompanyByKey(String key);

    void getAllPatentIdWithoutPdfs() throws IOException;

    Result clearPatentByNo(String no);
}
