import pymysql
from config.settings import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

def init_database():
    """
    데이터베이스 테이블 초기화
    - 앱 시작 시 자동으로 테이블 생성
    - 이미 존재하면 건너뜀
    """
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT,
        autocommit=False
    )
    
    try:
        with conn.cursor() as cursor:
            # 사용자 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    phoneNum VARCHAR(20) NOT NULL,
                    accountId VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    measure_cnt INT DEFAULT 0,
                    UNIQUE KEY unique_name_phone (name, phoneNum)
                )
            """)
            
            # 측정 결과 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS measurements (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    image_url VARCHAR(500) NOT NULL,
                    leftover_ratio FLOAT NOT NULL,
                    measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    INDEX idx_user_id (user_id),
                    INDEX idx_measured_at (measured_at)
                )
            """)
        
        conn.commit()
        print("✅ 데이터베이스 테이블이 성공적으로 초기화되었습니다.")
    except Exception as e:
        conn.rollback()
        print(f"❌ 데이터베이스 초기화 실패: {str(e)}")
        raise
    finally:
        conn.close()