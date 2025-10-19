from datetime import timedelta
import pymysql
from pymysql.cursors import DictCursor
from typing import List, Dict, Optional

from config.settings import (
    DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from services.auth_service import create_access_token

class DatabaseManager:
    """데이터베이스 CRUD 작업 관리 클래스"""
    
    def __init__(self):
        """데이터베이스 연결 설정"""
        self.db_config = {
            'host': DB_HOST,
            'user': DB_USER,
            'password': DB_PASSWORD,
            'database': DB_NAME,
            'port': DB_PORT,
            'cursorclass': DictCursor,
            'autocommit': False
        }
    
    def get_connection(self):
        """데이터베이스 연결 생성"""
        return pymysql.connect(**self.db_config)
    
    def create_user(self, name: str, phone_num: str, account_id: str) -> int:
        """
        사용자 생성
        
        Args:
            name: 사용자 이름
            phone_num: 전화번호
            account_id: 계정 ID
        
        Returns:
            생성된 사용자 ID
        
        Raises:
            ValueError: 이미 등록된 이름+전화번호 조합인 경우
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                # 이름-전화번호 쌍 중복 체크
                cursor.execute(
                    "SELECT id FROM users WHERE name = %s AND phoneNum = %s", 
                    (name, phone_num)
                )
                if cursor.fetchone():
                    raise ValueError("이미 등록된 이름과 전화번호 조합입니다.")
                
                # 사용자 등록
                cursor.execute(
                    "INSERT INTO users (name, phoneNum, accountId) VALUES (%s, %s, %s)",
                    (name, phone_num, account_id)
                )
                user_id = cursor.lastrowid
            conn.commit()
            return user_id
        except ValueError:
            raise
        except Exception as e:
            conn.rollback()
            raise Exception(f"사용자 생성 실패: {str(e)}")
        finally:
            conn.close()
    
    def login_user(self, phone_num: str) -> str:
        """
        사용자 로그인 (전화번호만)
        - 동일 전화번호로 여러 사용자가 있을 수 있으므로 첫 번째 사용자로 로그인
        
        Args:
            phone_num: 전화번호
        
        Returns:
            JWT 액세스 토큰
        
        Raises:
            ValueError: 등록되지 않은 전화번호인 경우
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, name FROM users WHERE phoneNum = %s LIMIT 1",
                    (phone_num,)
                )
                user = cursor.fetchone()
                
                if not user:
                    raise ValueError("등록되지 않은 전화번호입니다.")
                
                # JWT 토큰 생성
                access_token = create_access_token(
                    data={"user_id": user['id']},
                    expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
                )
                return access_token
        finally:
            conn.close()
    
    def login_user_with_name(self, name: str, phone_num: str) -> str:
        """
        사용자 로그인 (이름 + 전화번호)
        
        Args:
            name: 사용자 이름
            phone_num: 전화번호
        
        Returns:
            JWT 액세스 토큰
        
        Raises:
            ValueError: 등록되지 않은 사용자인 경우
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, name FROM users WHERE name = %s AND phoneNum = %s",
                    (name, phone_num)
                )
                user = cursor.fetchone()
                
                if not user:
                    raise ValueError("등록되지 않은 사용자입니다.")
                
                # JWT 토큰 생성
                access_token = create_access_token(
                    data={"user_id": user['id']},
                    expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
                )
                return access_token
        finally:
            conn.close()
    
    def get_user_info(self, user_id: int) -> Optional[Dict]:
        """
        사용자 정보 조회
        
        Args:
            user_id: 사용자 ID
        
        Returns:
            사용자 정보 딕셔너리 (없으면 None)
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, name, phoneNum, accountId, measure_cnt, created_at FROM users WHERE id = %s",
                    (user_id,)
                )
                return cursor.fetchone()
        finally:
            conn.close()
    
    def save_measurement(self, user_id: int, image_url: str, leftover_ratio: float) -> int:
        """
        측정 결과 저장 및 측정 횟수 증가
        
        Args:
            user_id: 사용자 ID
            image_url: S3 이미지 URL
            leftover_ratio: 잔반 비율
        
        Returns:
            저장된 측정 ID
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                # 측정 결과 저장
                cursor.execute(
                    """INSERT INTO measurements (user_id, image_url, leftover_ratio) 
                       VALUES (%s, %s, %s)""",
                    (user_id, image_url, leftover_ratio)
                )
                measurement_id = cursor.lastrowid
                
                # 사용자 측정 횟수 증가
                cursor.execute(
                    "UPDATE users SET measure_cnt = measure_cnt + 1 WHERE id = %s",
                    (user_id,)
                )
            conn.commit()
            return measurement_id
        except Exception as e:
            conn.rollback()
            raise Exception(f"측정 결과 저장 실패: {str(e)}")
        finally:
            conn.close()
    
    def get_user_history(self, user_id: int, limit: int = 50) -> List[Dict]:
        """
        사용자의 측정 이력 조회
        
        Args:
            user_id: 사용자 ID
            limit: 조회할 최대 개수 (기본 50개)
        
        Returns:
            측정 이력 리스트
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT id, image_url, leftover_ratio, measured_at 
                       FROM measurements 
                       WHERE user_id = %s 
                       ORDER BY measured_at DESC 
                       LIMIT %s""",
                    (user_id, limit)
                )
                return cursor.fetchall()
        finally:
            conn.close()
    
    def delete_measurement(self, measurement_id: int, user_id: int) -> bool:
        """
        측정 결과 삭제 (본인 것만 가능)
        
        Args:
            measurement_id: 측정 ID
            user_id: 사용자 ID
        
        Returns:
            삭제 성공 여부
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                # 소유자 확인 및 삭제
                cursor.execute(
                    "DELETE FROM measurements WHERE id = %s AND user_id = %s",
                    (measurement_id, user_id)
                )
                deleted = cursor.rowcount > 0
                
                # 삭제 성공 시 측정 횟수 감소
                if deleted:
                    cursor.execute(
                        "UPDATE users SET measure_cnt = GREATEST(measure_cnt - 1, 0) WHERE id = %s",
                        (user_id,)
                    )
            conn.commit()
            return deleted
        except Exception as e:
            conn.rollback()
            raise Exception(f"측정 결과 삭제 실패: {str(e)}")
        finally:
            conn.close()