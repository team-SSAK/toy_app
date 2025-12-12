import uuid
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException

from config.settings import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, S3_BUCKET_NAME

class S3Service:
    """AWS S3 이미지 업로드/삭제 서비스"""
    
    def __init__(self):
        """S3 클라이언트 초기화"""
        self.client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
        self.bucket_name = S3_BUCKET_NAME
    
    def upload_image(self, image_data: bytes, filename: str = None) -> str:
        """
        S3에 이미지 업로드 후 URL 반환
        
        Args:
            image_data: 이미지 바이너리 데이터
            filename: 파일명 (기본값: UUID 자동 생성)
        
        Returns:
            업로드된 이미지의 공개 URL
        
        Raises:
            HTTPException: S3 업로드 실패 시
        """
        if filename is None:
            filename = f"{uuid.uuid4()}.jpg"
        
        try:
            # 날짜별 폴더 구조로 저장 (예: leftover-images/2025/01/15/abc.jpg)
            s3_key = f"leftover-images/{datetime.now().strftime('%Y/%m/%d')}/{filename}"
            
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=image_data,
                ContentType='image/jpeg'
            )
            
            # 공개 URL 생성
            image_url = f"https://{self.bucket_name}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
            return image_url
        except ClientError as e:
            raise HTTPException(status_code=500, detail=f"S3 업로드 실패: {str(e)}")
    
    def delete_image(self, image_url: str) -> bool:
        """
        S3에서 이미지 삭제
        
        Args:
            image_url: 삭제할 이미지의 전체 URL
        
        Returns:
            삭제 성공 여부
        """
        try:
            # URL에서 S3 키 추출
            # 예: https://bucket.s3.region.amazonaws.com/path/to/file.jpg -> path/to/file.jpg
            s3_key = image_url.split(f"{self.bucket_name}.s3.{AWS_REGION}.amazonaws.com/")[1]
            
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return True
        except Exception as e:
            print(f"S3 삭제 실패: {str(e)}")
            return False