# CONTRIBUTING

<br>

### 1. 필자 등록

1. `_authors` 폴더에 필자 정보 파일 추가하기
    - 파일 이름: `lastname.firstname.md`
1. 파일 상단에 [front matter] 작성
    ```md
    ---
    name: 
    title: 
    image:
    ---
    ```
    - name
        - `lastname.firstname`
        - post의 author와 매칭
        - 특수 문자 없이 영소문자, .(점) 만 사용
    - title
        - 한글 이름
    - image: 
        - `/graph/files/authors/`의 이미지 파일 경로
        - 필수 아님

<br>

### 2. 태그 등록 (option)

1. `_tags` 디렉토리에 `tag-name.md` 이름으로 태그 정보 파일 추가
1. 파일 상단에 [front matter] 작성
    ```md
    ---
    name: 
    title:
    --- 
    ```
    - name
        - `tag-name`
        - post의 tags 배열의 항목과 매칭
        - 특수 문자 없이 영소문자, 숫자, -(하이픈), .(점) 만 사용
    - title: 
        - 좀 더 길고 구체적인 설명

<br>

### 3. 글 작성

1. 처음 글을 쓰는 필자이라면 **글쓴이 등록** (필수)
1. 태그가 새로 등장했다면 **태그 등록** (선택)
1. `_posts` 폴더에 파일 생성
    - 파일 이름: `yyyy-mm-dd-slug.md` 이름으로 파일 추가
    - gfm 문법, kramdown 파서, rouge 문법 강조기 사용
    - slug
        - 해당 포스트의 고유 키로 url의 일부로 사용
        - 특수 문자 없이 영소문자, 숫자, -(하이픈) 만 사용
    - yyyy-mm-dd
        - 발행 년, 월, 일
1. 파일 상단에 [front matter] 작성
    ```md
    ---
    layout: post
    use_math: true

    title: 'The PageRank Citation Ranking: Bringing Order to the Web'
    author: Sumin.Han
    date: 2022-03-25 13:03
    tags: [pagerank]
    ---
    ```
    - layout: post
    - use_math:
        - 수식을 적어야하면 true, 아니면 false
    - title:
        - 제목
    - author: `lastname.firstname`
        - 필자
    - date: `YYYY-MM-DD HH:MM:SS` 
        - 발행일
    - tags: `[tag1, tag2, tag3,...]`
        - 태그 목록 (선택)
1. 글에서 사용할 이미지는 `/graph/files/posts/{논문-제목}/` 디렉토리에 넣어서, 사용