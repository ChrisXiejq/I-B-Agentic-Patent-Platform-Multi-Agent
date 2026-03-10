package com.inovationbehavior.backend.util;

// COSConnector.java
import com.qcloud.cos.*;
import com.qcloud.cos.auth.*;
import com.qcloud.cos.model.*;
import com.qcloud.cos.region.*;
import com.qcloud.cos.exception.*;

import java.io.File;
import java.util.TreeMap;

public class PutFile {
    String tmpSecretId = "";   // 替换为 STS 接口返回给您的临时 SecretId
    String tmpSecretKey = "";  // 替换为 STS 接口返回给您的临时 SecretKey
    String sessionToken = "";  // 替换为 STS 接口返回给您的临时 Token

    public PutFile(String path) {
        this.path = path;
    }

    String path = "";

    public void PutFileToCOS(TreeMap<String, Object> credentials, File file,String filename){
        tmpSecretId = (String) credentials.get("tmpSecretId");
        tmpSecretKey = (String) credentials.get("tmpSecretKey");
        sessionToken = (String) credentials.get("sessionToken");

        COSCredentials cred = new BasicSessionCredentials(tmpSecretId, tmpSecretKey, sessionToken);
        Region region = new Region("ap-shanghai");
        ClientConfig clientConfig = new ClientConfig(region);
        COSClient cosClient = new COSClient(cred, clientConfig);

        String key = path + "/"+filename;
        String bucketName = "inovation-behavior-1324093188";

        PutObjectRequest putObjectRequest = new PutObjectRequest(bucketName, key, file);

        // 设置 x-cos-security-token header 字段
        ObjectMetadata objectMetadata = new ObjectMetadata();
        objectMetadata.setSecurityToken(sessionToken);
        putObjectRequest.setMetadata(objectMetadata);


        try {
            PutObjectResult putObjectResult = cosClient.putObject(putObjectRequest);
            // 成功：putobjectResult 会返回文件的 etag
            String etag = putObjectResult.getETag();
        } catch (CosServiceException e) {
            //失败，抛出 CosServiceException
            e.printStackTrace();
        } catch (CosClientException e) {
            //失败，抛出 CosClientException
            e.printStackTrace();
        }

        // 记得最后关闭COSClient
        cosClient.shutdown();
    }
}

