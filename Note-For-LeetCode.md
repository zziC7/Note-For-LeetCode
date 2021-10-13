# Note of LeetCode 

### **2021.4.13** 二叉搜索树

> #### LeetCode 783-二叉搜索树节点最小距离 
>
> https://leetcode-cn.com/problems/minimum-distance-between-bst-nodes/
>
> #### ***二叉搜索树***
>
> ​	二叉搜索树(Binary Search Tree，简写BST)，又称为二叉排序树，属于树的一种，通过二叉树将数据组织起来，树的每个节点都包含了健值 key、数据值 data、左子节点指针、右子节点指针。其中键值 key 是最核心的部分，它的值决定了树的组织形状；数据值 data 是该节点对应的数据，有些场景可以忽略，举个例子，key 为身份证号而 data 为人名，通过身份证号找人名；左子节点指针指向左子节点；右子节点指针指向右子节点。
>
> #### ***二叉搜索树特点***
>
> - 左右子树也分别是二叉搜索树
> - 左子树的所有节点 key 值都小于它的根节点的 key 值
> - 右子树的所有节点 key 值都大于他的根节点的 key 值
> - 二叉搜索树可以为一棵空树
> - 一般来说，树中的每个节点的 key 值都不相等，但根据需要也可以将相同的 key 值插入树中

<img src="https://assets.leetcode.com/uploads/2021/02/05/bst1.jpg" alt="img"  />

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    void BST2Vec(TreeNode* root, vector<int>& vec)
    {
        //将二叉搜索树转换为数组
        if(!root) return;
        BST2Vec(root->left,vec); //左
        vec.push_back(root->val); //中
        BST2Vec(root->right,vec); //右
    }

    int minDiffInBST(TreeNode* root) {
        vector<int> vec;
        BST2Vec(root,vec);
        int minDiff = INT_MAX;
        for(int i=0; i<vec.size()-1; ++i)
        {
            minDiff = min(minDiff,vec[i+1]-vec[i]);
        }
        return minDiff;
    }
};
```



### 2021.4.14 **前缀树(字典树) **

> #### **LeetCode 208-Trie **
>
> https://leetcode-cn.com/problems/implement-trie-prefix-tree/solution/fu-xue-ming-zhu-cong-er-cha-shu-shuo-qi-628gs/

Trie（发音类似 "try"）或者说 前缀树 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。

```c++
class TrieNode{ 
    //前缀树节点
public:
    bool isEnd; //是否到一个串的末尾
    vector<TrieNode*> children;  //子节点
    TrieNode(): isEnd(false), children(26,nullptr){} //构造函数
    ~TrieNode(){  
        //析构
        for(auto& c:children)  delete c; //引用型自动推导，是指针传递而非值传递
    }
};

class Trie {
private:
    TrieNode* root; //根节点
public:
    /** Initialize your data structure here. */
    Trie() {
        root = new TrieNode();
    }
    
    /** Inserts a word into the trie. */
    void insert(string word) {
        TrieNode* p = root;
        for(char ch : word){
            int i = ch - 'a'; //得到数组下标
            if(!p->children[i]) p->children[i] = new TrieNode(); //若这个子节点还没有创建，就new一个
            p = p->children[i]; //接着往下遍历
        }
        p->isEnd = true;  //遍历完一个字符串，在最后一个节点将isEnd设置为TRUE,代表一个字符串的结束
    }
    
    /** Returns if the word is in the trie. */
    bool search(string word) {
        TrieNode* p = root;
        for(char ch : word){
            int i = ch - 'a'; //得到数组下标
            if(!p->children[i]) return false; //没有这个字符的子节点，说明该字符串不存在
            p = p->children[i]; //接着往下遍历
        }
        return p->isEnd; //遍历完了，由最后这个节点的isEnd判断这是不是完整的字符串
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
         TrieNode* p = root;
        for(char ch : prefix){
            int i = ch - 'a'; //得到数组下标
            if(!p->children[i]) return false; //没有这个字符的子节点，说明由该字符串开头的字符串不存在
            p = p->children[i]; //接着往下遍历
        }
        return true; //遍历完了。与查找不同，不管isEnd是True还是False,都已经确定了有这个子串开头的字符串，所以返回true
    }
};
```



### 2021.4.15  动态规划 

> LeetCode 198-HouseRobber 
>
> https://leetcode-cn.com/problems/house-robber/solution/dong-tai-gui-hua-jie-ti-si-bu-zou-xiang-jie-cjavap/

动态规划的的四个解题步骤是：

1. 定义子问题
2. 写出子问题的递推关系
3. 确定 DP 数组的计算顺序
4. 空间优化（可选）

```c++
int rob(vector<int>& nums) {
    if (nums.size() == 0) {
        return 0;
    }
    // 子问题：
    // f(k) = 偷 [0..k) 房间中的最大金额

    // f(0) = 0
    // f(1) = nums[0]
    // f(k) = max{ rob(k-1), nums[k-1] + rob(k-2) }

    int N = nums.size();
    vector<int> dp(N+1, 0);
    dp[0] = 0;
    dp[1] = nums[0];
    for (int k = 2; k <= N; k++) {
        dp[k] = max(dp[k-1], nums[k-1] + dp[k-2]);
    }
    return dp[N];
}
```

空间优化： 空间复杂度： O(n) --> O(1)

```c++
int rob(vector<int>& nums) {
    int prev = 0;
    int curr = 0;

    // 每次循环，计算“偷到当前房子为止的最大金额”
    for (int i : nums) {
        // 循环开始时，curr 表示 dp[k-1]，prev 表示 dp[k-2]
        // dp[k] = max{ dp[k-1], dp[k-2] + i }
        int temp = max(curr, prev + i);
        prev = curr;
        curr = temp;
        // 循环结束时，curr 表示 dp[k]，prev 表示 dp[k-1]
    }

    return curr;
}
```

### 2021.4.18 快速排序

> https://blog.csdn.net/starzyh/article/details/90272347
>
> - ### 1.快排的实现逻辑：
>
>   - 先从数列中取出一个数作为基准数(通常取第一个数)。
>   - 分区过程，将比这个数大的数全放到它的右边，小于或等于它的数全放到它的左边。
>   - 再对左右区间重复第二步，直到各区间只有一个数。
>
>   ### 2.示意图
>
>   <img src="https://img-blog.csdnimg.cn/20190517164201640.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3N0YXJ6eWg=,size_16,color_FFFFFF,t_70" alt="img" style="zoom: 67%;" />
>
>   ### 3.代码
>
>   ```c++
>   void QuickSort(vector<int>& nums, int left, int right)
>   {
>   	if (left == right) return;
>   	if (left < right)
>   	{
>   		int pivot = nums[left];
>   		int low = left, high = right;
>   		while (low < high)
>   		{
>   			while (nums[high] >= pivot && low < high)
>   			{
>   				high--;
>   			}
>   			nums[low] = nums[high];
>                                                                                                                                                       
>   			while (nums[low] <= pivot && low < high)
>   			{
>   				low++;
>   			}
>   			nums[high] = nums[low];
>   		}
>   		nums[low] = pivot;
>   		QuickSort(nums, left, low - 1);
>   		QuickSort(nums, low + 1, right);
>   	}
>   }
>   ```



### 2021.4.19 哈希表

> https://blog.csdn.net/gp1330782530/article/details/106413778 unordered_set的介绍及使用
>
> https://leetcode-cn.com/problems/linked-list-cycle-ii/LeetCode-142 环形链表
>
> 哈希表最大的优点，就是**把数据的存储和查找消耗的时间大大降低，几乎可以看成是常数时间；而代价仅仅是消耗比较多的内存**。然而在当前可利用内存越来越多的情况下，用空间换时间的做法是值得的。另外，编码比较容易也是它的特点之一。
>
> ```c++
>  ListNode *detectCycle(ListNode *head) {
>         unordered_set<ListNode*> visited; //哈希表
>         while(head){
>             if(visited.count(head))return head; //如果找到重复元素，说明有环，且这个节点就是环的起点
> 
>             visited.insert(head); //如果没有这个元素，就加入到哈希表里
>             head = head->next; // 继续向下遍历
>         }
>         return nullptr;
>     }
> ```
>
> 时间复杂度：O(N)，需要遍历链表中的每个节点
>
> 空间复杂度：O(N)，需要将每个节点存储到哈希表中
>
> *利用快慢指针可以将空间复杂度O(N) --> O(1) （追及问题） 这里略，详见LeetCode官方题解



### 2021.4.21  动态规划 

> LeetCode 91-解码方法
>
> https://leetcode-cn.com/problems/decode-ways/

```c++
int numDecodings(string s) {
	int n = s.size();
	vector<int> f(n + 1);  //代表长度为n的字符串的解码方法总数
	f[0] = 1; //初始条件：空字符串，一种解码方法
	for (int i = 0; i < n; ++i) {
		//判断末尾字符是否能解码（是不是0）
		if (s[i] != '0') {
			f[i + 1] += f[i];
		}
		//判断末尾2位是否能解码
		if (i > 0 && s[i - 1] != '0' && ((s[i - 1] - '0') * 10 + (s[i] - '0') <= 26)) {
			f[i + 1] += f[i - 1];
		}
	}
	return f[n];
}
```

时间复杂度：O(n)，遍历长度为n的字符串

空间复杂度：O(n),   创建大小为n+1的数组存放长度为n的字符串的解码方法总数



**空间优化**：O(n) --> O(1)注意到在状态转移方程中，f<sub>i</sub>的值仅与和 f<sub>i-1</sub>和f<sub>i-2</sub>有关，

因此我们可以使用三个变量进行状态转移，省去数组的空间。

```c++
int numDecodings(string s) {
	int n = s.size();
	// a = f[i-1], b = f[i], c = f[i+1]
	int a = 0, b = 1, c;
	//初始条件：空字符串，一种解码方法
	for (int i = 0; i < n; ++i) {
		c = 0; //每次进入循环都重置c为0
		//判断末尾字符是否能解码（是不是0）
		if (s[i] != '0') {
			c += b;
		}
		//判断末尾2位是否能解码
		if (i > 0 && s[i - 1] != '0' && ((s[i - 1] - '0') * 10 + (s[i] - '0') <= 26)) {
			c += a;
		}
        //往前挪一步
		a = b;
		b = c;
	}
	return c;
}
```



### 2021.4.24 动态规划

> LeetCode-377 组合总和Ⅳ https://leetcode-cn.com/problems/combination-sum-iv/

```c++
int combinationSum4(vector<int>& nums, int target) {
        vector<int> dp(target + 1);
        dp[0] = 1;
        for (int i = 1; i <= target; i++) {
            for (int& num : nums) {
                if (num <= i && dp[i - num] < INT_MAX - dp[i]) {
                    dp[i] += dp[i - num];
                }
            }
        }
        return dp[target];
    }
```



### 2021.4.25 二叉搜索树

> LeetCode-897 递增顺序搜索树 https://leetcode-cn.com/problems/increasing-order-search-tree/
>
> 给你一棵二叉搜索树，请你 **按中序遍历** 将其重新排列为一棵递增顺序搜索树，使树中最左边的节点成为树的根节点，并且每个节点没有左子节点，只有一个右子节点。

```c++
    void inorder(TreeNode *node, vector<int> &res) {
            if (node == nullptr) {
                return;
            }
            inorder(node->left, res);
            res.push_back(node->val);
            inorder(node->right, res);
        }

    TreeNode *increasingBST(TreeNode *root) {
        vector<int> res;
        inorder(root, res);

        TreeNode *dummyNode = new TreeNode(-1);
        TreeNode *currNode = dummyNode;
        for (int value : res) {
            currNode->right = new TreeNode(value);
            currNode = currNode->right;
        }
        return dummyNode->right;
    }
```



### 2021.4.26 二分查找的应用

> LeetCode-1011 在D天送达包裹的能力 https://leetcode-cn.com/problems/capacity-to-ship-packages-within-d-days/
>
> **二分查找算法的注意事项：**
>
> **1.循环退出条件：**
>
> 注意是low<=high
>
> `while(left <= right)` 的终止条件是 `left == right + 1`，写成区间的形式就是 `[right + 1, right]`，或者带个具体的数字进去 `[3, 2]`，可见这时候区间为空，因为没有数字既大于等于 3 又小于等于 2 的吧。所以这时候 while 循环终止是正确的，直接返回 -1 即可。
>
> `while(left < right)` 的终止条件是 `left == right`，写成区间的形式就是 `[left, right]`，或者带个具体的数字进去 `[2, 2]`，这时候区间非空，还有一个数 2，但此时 while 循环终止了。也就是说这区间 `[2, 2]` 被漏掉了，索引 2 没有被搜索，如果这时候直接返回 -1 就是错误的。
>
> **2.mid取值**
>
> 实际上，mid=(low+high)/2 这种写法是有问题的。因为如果 low 和 high 比较大的话，两者之和就有可能会溢出。改进的方法是将 mid 的计算方式写成low+(high-low)/2。更进一步，如果要将性能优化到极致的话，我们可以将这里的除以 2 操作转化成位运算 low+((high-low)>>1)。因为相比除法运算来说，计算机处理位运算要快得多。
>
> **3.low 和 high 的更新**
>
> low=mid+1，high=mid-1。注意这里的 +1 和 -1，如果直接写成 low=mid 或者 high=mid，就可能会发生死循环。比如，当 high=3，low=3 时，如果 a[3]不等于 value，就会导致一直循环不退出。

```c++
   class Solution {
public:
        bool check(vector<int> &weights, int D, int limit) {
        int cnt = 1; //记录该承载力下运送完所有包裹需要的天数
        int cur = 0; //记录现在这艘船这一天已经运送包裹的重量
        for(auto &weight : weights) {
            //如果运这个包裹会超载，那今天就运到这里，天数+1，重量清零
            if(cur + weight > limit) {
                cnt++;
                cur = 0;
            }
            if (cnt > D)return false; //当前天数已经超出要求，直接返回false
            cur += weight; //把当前这个包裹的重量记录到cur里
        }
        return cnt <= D; //运送完了，判断所需天数是否小于D
    }

    int shipWithinDays(vector<int>& weights, int D) {
        //二分查找
        int left = *max_element(weights.begin(), weights.end()); //最低运载能力：最大包裹的重量
        int right = accumulate(weights.begin(),weights.end(),0);//最大运载能力：所有包裹的重量和
        int ans = right;

        while(left<=right)
        {
            int mid = (left + right) /2;
            if(check(weights,D,mid))
            {
                right = mid - 1;
            }
            else left = mid + 1; //注意，这里一定return的是else里面的
        }
        return left;
    }
};
```



### 2021.4.27 二叉搜索树&dfs

> LeetCode-938 二叉搜索树范围和 https://leetcode-cn.com/problems/range-sum-of-bst/
>
> 当时做的时候没有多想，直接中序遍历转数组了，时间和空间效率都比较差

> #### 思路：
>
> **1、root 节点为空**
>
> 返回 0。
>
> **2、root 节点的值大于high**
>
> 由于二叉搜索树右子树上所有节点的值均大于根节点的值，即均大于high，故无需考虑右子树，返回左子树的范围和。
>
> **3、root 节点的值小于low**
>
> 由于二叉搜索树左子树上所有节点的值均小于根节点的值，即均小于low，故无需考虑左子树，返回右子树的范围和。
>
>  **4、root 节点的值在[low, high]范围内**
>
> 此时应返回 root 节点的值、左子树的范围和、右子树的范围和这三者之和。

```c++
class Solution {
public:
    int rangeSumBST(TreeNode *root, int low, int high) {
        if (root == nullptr) {
            return 0;
        }
        if (root->val > high) {
            return rangeSumBST(root->left, low, high);
        }
        if (root->val < low) {
            return rangeSumBST(root->right, low, high);
        }
        return root->val + rangeSumBST(root->left, low, high) + rangeSumBST(root->right, low, high);
    }
};
```

**复杂度分析**

- 时间复杂度：O*(*n)，其中n是二叉搜索树的节点数。
- 空间复杂度：O*(*n)，空间复杂度主要取决于栈空间的开销。



### 2021.4.30 哈希表

> LeetCode - 137 只出现一次的数字II   https://leetcode-cn.com/problems/single-number-ii/

```c++
int singleNumber(vector<int>& nums) {
	unordered_map<int, int> freq;
	for (int num : nums) {
		++freq[num];
	}
	int ans;
	for (auto [num, occ] : freq) {
		if (occ == 1) {
			ans = num;
			break;
		}
	}
	return ans;
}
```



### 2021.5.1 哈希表&dfs

> LeetCode - 690 员工的重要性  https://leetcode-cn.com/problems/employee-importance/

```c++
unordered_map<int, Employee*> mp; //定义哈希表，存储每一个员工及其id

int dfs(int id) {
    Employee* employee = mp[id];
    int total = employee->importance;
    //把每一个下属的重要度都加到total里
    for (int i : employee->subordinates) {
        total += dfs(i);
    }
    return total;
}


int getImportance(vector<Employee*> employees, int id) {
    //将employees读取到哈希表里
    for (auto& employee : employees) {
        mp[employee->id] = employee;
    }
    return dfs(id);
}
```



### 2021.5.10 dfs&二叉树叶子

> LeetCode - 872 叶子相似的树 https://leetcode-cn.com/problems/leaf-similar-trees/

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    void dfs(TreeNode* root, vector<int>& vec){
        if(!root->left && !root->right){
            vec.push_back(root->val); //检测到叶节点，加入到数组里
        }
        else{  //不是叶节点
            if(root->left)  dfs(root->left, vec); //左子树下面还有叶节点，往下搜索
            if(root->right) dfs(root->right, vec); //右子树下面还有叶节点，往下搜索
        }
    }

    bool leafSimilar(TreeNode* root1, TreeNode* root2) {
        vector<int> vec1;
        if(root1) dfs(root1, vec1);

        vector<int> vec2;
        if(root2) dfs(root2, vec2);

        return vec1 == vec2;
    }
};
```



### 2021.5.17 二叉树&dfs+bfs

> LeetCode - 993 二叉树的堂兄弟节点 https://leetcode-cn.com/problems/cousins-in-binary-tree/

1.dfs 主要是利用dfs找到x和y,再比较x,y的深度和其父节点是否相同

```c++
class Solution {
private:
	int x_depth = 0;
	int y_depth = 0;
	TreeNode* x_father;
	TreeNode* y_father;
public:
    bool isCousins(TreeNode* root, int x, int y) {
        dfs(root,x,y,0);
        return x_depth == y_depth && x_father != y_father;
    }
	void dfs(TreeNode* root, int x, int y, int depth) {
		if(root == nullptr) return ;
		if(root->left != nullptr) {
			if(root->left->val == x) {
				x_depth = depth+1;
				x_father = root;
			}
			if(root->left->val == y) {
				y_depth = depth+1;
				y_father = root;
			}
			dfs(root->left,x,y,depth+1);
		}
		if(root->right != nullptr) {
			if(root->right->val == x) {
				x_depth = depth+1;
				x_father = root;
			}
			if(root->right->val == y) {
				y_depth = depth+1;
				y_father = root;
			}
			dfs(root->right,x,y,depth+1);
		}
	}
};
```

2.bfs 

我们使用的是层序遍历，如果每次遍历一层，那么这一层的元素的深度是相同的。

在确定深度上面，BFS一直很可以的。

**因此我们在每一层，看看是否有出现 x 和 y，其中分为以下三种情况：**

1.  x 和 y 都没出现 → 那只能往下一层继续找了

2.  x 和 y 只出现一个 → 两个元素的深度不同，不可能为兄弟，返回false

3.  x 和 y 都出现了，好耶，但是还不够好

   - `x` 和 `y` 父节点相同 → 不是堂兄弟，是亲兄弟，不行，返回`false`
   - `x` 和 `y` 父节点不同 → 满足题目条件了，好耶，返回`true`	

   众所周知，BFS需要用到队列，那么我们应该如何设计队列的数据类型？
   在这里，我采用了 **pair<TreeNode*, TreeNode*>（其实pair<TreeNode*, int>也可以），**

   其中pair中，第一个元素记录指向当前结点的指针，第二个元素记录指向当前结点的父结点的指针，这样就可以完美应对上面所说的三种情况了。

```c++
class Solution {
public:
    using PTT = pair<TreeNode*, TreeNode*>;
    bool isCousins(TreeNode* root, int x, int y) {
        // 使用队列q来进行bfs
        // 其中pair中，p.first 记录当前结点的指针，p.second 记录当前结点的父结点的指针
        queue<PTT> q;
        q.push({root, nullptr});
        while(!q.empty()) {
            int n = q.size();
            vector<TreeNode*> rec_parent;
            for(int i = 0; i < n; i++) {
                auto [cur, parent] = q.front(); q.pop();
                if(cur->val == x || cur->val == y)
                    rec_parent.push_back(parent);
                if(cur->left) q.push({cur->left, cur});
                if(cur->right) q.push({cur->right, cur});
            }
            // `x` 和 `y` 都没出现
            if(rec_parent.size() == 0)
                continue;
            // `x` 和 `y` 只出现一个
            else if(rec_parent.size() == 1)
                return false;
            // `x` 和 `y` 都出现了
            else if(rec_parent.size() == 2)
                // `x` 和 `y` 父节点 相同/不相同 ？
                return rec_parent[0] != rec_parent[1];
        }
        return false;
    }
};
```



### 2021.5.21 dp -- 最长公共子串

> LeetCode 1035 不相交的线 https://leetcode-cn.com/problems/uncrossed-lines/

![image-20210521111506583](C:\Users\86188\AppData\Roaming\Typora\typora-user-images\image-20210521111506583.png)

```c++
class Solution {
public:
    int maxUncrossedLines(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size(), n = nums2.size();
        int dp[m+1][n+1];
        memset(dp, 0, sizeof(dp));
        for(int i = 1; i <= m; i++) {
            for(int j = 1; j <= n; j++) {
                if(nums1[i-1] == nums2[j-1])
                    dp[i][j] = dp[i-1][j-1] + 1;
                else
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
        return dp[m][n];    
    }
};
```



### 2021.5.27 汉明距离

> LeetCode - 461 汉明距离 https://leetcode-cn.com/problems/hamming-distance/

两个整数之间的[汉明距离](https://baike.baidu.com/item/汉明距离)指的是这两个数字对应二进制位不同的位置的数目。

**1、直接使用内置计数功能**

```c++
class Solution {
public:
    int hammingDistance(int x, int y) {
        return __builtin_popcount(x ^ y);
    }
};
```

**2、移位统计1的数量**

```c++
class Solution {
public:
    int hammingDistance(int x, int y) {
        int s = x ^ y, ret = 0;
        while (s) {
            ret += s & 1;
            s >>= 1;
        }
        return ret;
    }
};
```

**3、Brian Kernighan 算法**

跳过所有的0，直接对1进行计数：令f(x) = x & (x-1) 就会删除x最右侧的1

<img src="https://assets.leetcode-cn.com/solution-static/461/3.png" alt="fig3" style="zoom: 25%;" />

```c++
class Solution {
public:
    int hammingDistance(int x, int y) {
        int s = x ^ y, ret = 0;
        while (s) {
            s &= s - 1;
            ret++;
        }
        return ret;
    }
};
```



### 2021.5.28 汉明距离总和

> LeetCode - 477 汉明距离总和 https://leetcode-cn.com/problems/total-hamming-distance/

在计算汉明距离时，我们考虑的是同一比特位上的值是否不同，而不同比特位之间是互不影响的。

对于数组 ***nums*** 中的某个元素 ***val***，若其二进制的第 ***i*** 位为 1，我们只需统计 ***nums*** 中有多少元素的第 ***i*** 位为 0，即计算出了 ***val***与其他元素在第 ***i*** 位上的汉明距离之和。

具体地，若长度为 ***n*** 的数组 ***nums*** 的所有元素二进制的第 ***i*** 位共有 ***c*** 个 ***1***，***n-c*** 个 ***0***，则些元素在二进制的第 ***i*** 位上的汉明距离之和为

$$
c*(n-c)
$$

```c++
class Solution {
public:
    int totalHammingDistance(vector<int> &nums) {
        int ans = 0, n = nums.size();
        for (int i = 0; i < 30; ++i) {
            int c = 0;
            for (int val : nums) {
                c += (val >> i) & 1; //统计这一位上1的个数
            }
            ans += c * (n - c); //加上第i位上的汉明距离和
        }
        return ans;
    }
};
```



### 2021.5.30 2的幂

> LeetCode - 231 2的幂   https://leetcode-cn.com/problems/power-of-two/
>
> 一个数 *n* 是 2 的幂，当且仅当 *n* 是正整数，并且 *n* 的二进制表示中仅包含 1 个 1。
>
> n & (n - 1) 指的是删掉最右侧的一个1

```c++
class Solution {
public:
    bool isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
};
```



### 2021.5.31 4的幂

> LeetCode - 342 4的幂   https://leetcode-cn.com/problems/power-of-four/

在2的幂的基础上，要保证这个数的二进制数只有一个1且这个1在奇数位上。

构造一个mask，使其与n合取，来检验是否所有的1都在奇数位上。

```c++
class Solution {
public:
    bool isPowerOfFour(int n) {
        return n > 0 && (n & (n - 1)) == 0 && (n & 0b10101010101010101010101010101010) == 0;
    }
};
```

```c++
class Solution {
public:
    bool isPowerOfFour(int n) {
        return n > 0 && (n & (n - 1)) == 0 && (n & 0xaaaaaaaa) == 0;
    }
};
```

另解：***利用以下性质***    

4的幂除以3余数一定为1，2的幂且不是4的幂除以3的余数一定为2

```c++
class Solution {
public:
    bool isPowerOfFour(int n) {
        return n > 0 && (n & (n - 1)) == 0 && n % 3 == 1;
    }
};
```



### 2021.6.1 前缀和&儿童节吃糖果

> LeetCode - 1744 吃最喜欢的糖果 https://leetcode-cn.com/problems/can-you-eat-your-favorite-candy-on-your-favorite-day/

```c++
class Solution {
public:
    vector<bool> canEat(vector<int>& candiesCount, vector<vector<int>>& queries) {
        int n = candiesCount.size();
        int m = queries.size();

        //前缀和
        vector<long> preSum(n);
        preSum[0] = candiesCount[0];
        for(int i=1; i<n; ++i) preSum[i] = preSum[i-1] + candiesCount[i];

        vector<bool> ans(m);
        for(int i=0; i<m; ++i)
        {
            int favoriteType = queries[i][0];
            long favoriteDay = queries[i][1];
            long dailyCap = queries[i][2];
            long minCandy = favoriteDay + 1;
            long maxCandy = (favoriteDay + 1) * dailyCap;
            // 理想区间是[day+1, (day+1)*dailyCap]
            if(favoriteType == 0 && preSum[favoriteType] >= minCandy) ans[i] = true; //type=0的特殊情况
            else if(favoriteType!=0 && preSum[favoriteType]>=minCandy && preSum[favoriteType-1]<maxCandy) ans[i] = true;
        }

        return ans;
    }
};
```



### 2021.6.2 前缀和&哈希表

> LeetCode - 523 连续的子数组和 https://leetcode-cn.com/problems/continuous-subarray-sum/

```c++
class Solution {
public:
    bool checkSubarraySum(vector<int>& nums, int k) {
        int m = nums.size();
        if (m < 2) {
            return false;
        }
        unordered_map<int, int> mp; //创建哈希表，记录每个余数出现的最小下标
        mp[0] = -1;
        int remainder = 0; //记录前缀和的余数
        for (int i = 0; i < m; i++) {
            remainder = (remainder + nums[i]) % k;
            if (mp.count(remainder)) {
                //此余数已经在哈希表里
                int prevIndex = mp[remainder]; //记录已经在哈希表里面的余数的下标
                if (i - prevIndex >= 2) {
                    return true; //下标差大于2，满足题目要求
                }
            } else {
                mp[remainder] = i; //此余数还未出现在哈希表中，加进去
            }
        }
        return false;
    }
};
```



### 2021.6.3 前缀和&哈希表

> LeetCode - 525 连续数组 https://leetcode-cn.com/problems/contiguous-array/

```c++
class Solution {
public:
    int findMaxLength(vector<int>& nums) {
        int maxLength = 0; //记录最大子数组长度
        unordered_map<int,int> mp; 
        mp[0] = -1;

        int counter = 0; //用于计数
        for(int i=0; i<nums.size(); ++i){
            if(nums[i]==1) counter++; //数组遍历到1，计数器+1
            else counter--; //数组遍历到0，计数器-1

            if(mp.count(counter)){
                //此数已经在哈希表中
                int prevIndex = mp[counter]; //记录上一次出现这个数的下标
                maxLength = max(maxLength, i-prevIndex); //取大
            }
            else mp[counter] = i; //这个数不在哈希表中，加进去
        }
        return maxLength;
    }
};
```



### 2021.6.4 相交链表

> LeetCode - 160 相交链表 https://leetcode-cn.com/problems/intersection-of-two-linked-lists/

一、哈希集合(自己写的)

```c++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if(!headA || !headB) return nullptr;
        unordered_set<ListNode*> st;
        ListNode *curr = headA;
        while(curr){
            st.insert(curr);
            curr=curr->next;
        }
        curr=headB;
        while(curr){
            if(st.count(curr))return curr;
            curr=curr->next;
        }
        return nullptr;
    }
};
```

***复杂度分析***

时间复杂度：O(m+n)，其中 m 和 n 是分别是链表A 和 B 的长度。需要遍历两个链表各一次。

空间复杂度：O(m)，其中 m 是链表 headA 的长度。需要使用哈希集合存储链表 headA 中的全部节点。



二、双指针 O(1) 难想到

```c++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if (headA == nullptr || headB == nullptr) {
            return nullptr;
        }
        ListNode *pA = headA, *pB = headB;
        while (pA != pB) {
            pA = pA == nullptr ? headB : pA->next;
            pB = pB == nullptr ? headA : pB->next;
        }
        return pA;
    }
};
```



### 2021.6.6 动态规划

> LeetCode - 474 一和零 https://leetcode-cn.com/problems/ones-and-zeroes/

```c++
class Solution {
public:
    int countChar(string str,char ch){
        int cnt = 0;
        for(char c:str){
            if(c==ch) cnt++;
        }
        return cnt;
    }

    int findMaxForm(vector<string>& strs, int m, int n) {
        int length = strs.size();
        //定义三维数组，dp[i][j][k]表示前i个字符串中用j个0和k和1子集的最大大小
        //最终答案应该是dp[length][m][n]
        vector<vector<vector<int>>> dp(length+1, vector<vector<int>>(m+1,vector<int>(n+1)));
        for(int i=1; i<=length; ++i){
            int zeroNum = countChar(strs[i-1],'0');
            int oneNum = countChar(strs[i-1],'1');
            for(int j=0; j<=m; ++j){
                for(int k=0; k<=n; ++k){
                    dp[i][j][k] = dp[i-1][j][k];
                    if(zeroNum<=j && oneNum<=k){
                        dp[i][j][k] = max(dp[i][j][k], dp[i-1][j-zeroNum][k-oneNum]+1);
                    }
                }
            }
        }
        return dp[length][m][n];
    }
};
```



### 2021.6.7 动态规划

> LeetCode - 494 目标和  https://leetcode-cn.com/problems/target-sum/

```c++
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        int sum = 0;
        for(int num:nums) sum += num;
        int diff = sum - target; //负数的个数为 (sum - target) / 2
        if(diff<0 || diff%2!=0) return 0; //不满足要求，直接返回0
        int n = nums.size();
        int neg = diff/2;
        vector<vector<int>> dp(n+1,vector<int>(neg+1)); //dp[i][j]表示在前i个数中找到和为j的方案数
        dp[0][0] = 1; //边界条件 dp[0][0]=1, dp[0][j] = 0(j!=0)
        for(int i=1; i<=n; ++i){
            int num = nums[i-1];
            for(int j=0; j<=neg; ++j){
                if(num>j) dp[i][j] = dp[i-1][j]; //当前num已经大于j，不能选
                else if(num<=j) dp[i][j] = dp[i-1][j] + dp[i-1][j-num];
            }
        }
        return dp[n][neg];
     }
};
```



### 2021.6.9 动态规划

> LeetCode 879 - 盈利计划**(阿里3.8面试原题)**    https://leetcode-cn.com/problems/profitable-schemes/

```c++
class Solution {
public:
    int profitableSchemes(int n, int minProfit, vector<int>& group, vector<int>& profit) {
        int len = group.size(), MOD = (int)1e9 + 7;
        //dp[i][j][k]表示前i个工作中(不一定全选)，选择j个员工，并保持至少k利润的方案数
        //最终答案是 dp[len][j][minProfit] j从0~n遍历
        vector<vector<vector<int>>> dp(len+1, vector<vector<int>>(n+1,vector<int>(minProfit+1)));
        dp[0][0][0] = 1;//边界条件
        for(int i=1; i<=len; ++i){   
            int members = group[i-1];
            int earn = profit[i-1];
            for(int j=0; j<=n; ++j){
                for(int k=0; k<=minProfit; ++k){
                    if(members>j){
                        //人数过多，这工作不能选
                        dp[i][j][k] = dp[i-1][j][k];
                    }else{
                        //这工作可以选
                        //使用max(0,k-earn)是因为题目要求利润至少为k而不是恰好为k
                        dp[i][j][k] = (dp[i-1][j][k] + dp[i-1][j-members][max(0,k-earn)]) % MOD;
                    }
                }
            }
        }
        int sum = 0;
        for(int j=0; j<=n; ++j){
            sum = (sum+dp[len][j][minProfit])%MOD;
        }
        return sum;
    }
};
```



### 2021.6.10 动态规划

> LeetCode - 518 零钱兑换Ⅱ      https://leetcode-cn.com/problems/coin-change-2/

```c++
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        vector<int> dp(amount+1); //dp[i]代表总金额为i的选择方案数
        dp[0] = 1;//边界条件，什么都不选才能使总金额为0
        for(int& coin:coins){
            for(int i=coin; i<=amount; ++i){
                dp[i] += dp[i-coin];
            }
        }
        return dp[amount];
    }
};
```



### 2021.6.11 动态规划

> LeetCode - 279 完全平方数  https://leetcode-cn.com/problems/perfect-squares/

```c++
class Solution {
public:
    int numSquares(int n) {
        vector<int> dp(n+1); //dp[i]表示组成和为i的完全平方数的个数最少
        dp[0] = 0; //组成0的个数为0
        for(int i=1; i<=n; ++i){
            int minn = INT_MAX; //记录所需最少完全平方数的个数(不算i在内)
            for(int j=1; j*j<=i; ++j){
                minn = min(minn, dp[i-j*j]); //遍历j找到最小的dp[i-j*j]
            }
            dp[i] = minn+1; //+1是因为还要加上i这个数
        }
        return dp[n];
    }
};
```



### 2021.6.22 回溯+剪枝

> LeetCode - 剑指Offer 38.字符串的排列 
>
> https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/

```c++
class Solution {
public:
    vector<string> res;

    void dfs(string s, int index){
        // 已经固定完所有字符，加入到结果集中
        if(index == s.size() - 1){
            res.push_back(s);
            return;
        }

        // 还未固定完所有字符
        set<char> st; // 用于储存已经固定的字符
        for(int i = index; i < s.size(); ++i){
            if(st.count(s[i])) continue; //该字符已经在这一位固定过，剪枝
            st.insert(s[i]);
            swap(s[i], s[index]); // 将s[i]固定到第 index 位
            dfs(s, index + 1);    // 继续向下固定第 index + 1 位
        }
    }

    vector<string> permutation(string s) {
        dfs(s, 0);
        return res;
    }
};
```



### 2021.6.23 哈希表(Hard)

> LeetCode - 149 直线上最多的点数 
>
> https://leetcode-cn.com/problems/max-points-on-a-line/

```c++
class Solution {
public:
    int maxPoints(vector<vector<int>>& points) {
        int n = points.size();
        // 小于3个点，必在一条直线上
        if(n < 3){
            return n;
        }

        int res = 2; // 至少有2个点在一条直线上
        for(int i = 0; i < n; ++i){
            unordered_map<double, int> mp; // 储存每个斜率的计数
            for(int j = 0; j < n; ++j){
                long long dx = points[j][0] - points[i][0];
                long long dy = points[j][1] - points[i][1];
                double slope = 1.0 * dy / dx;
                if(mp.count(slope)){
                    mp[slope]++;   // 该斜率已在哈希表中，计数+1
                }else{
                    mp[slope] = 2; // 该斜率还未在哈希表中，初始化为2
                }
                res = max(res, mp[slope]);
            } 
        }
        return res;
    }
};
```



### 2021.7.3 字符频率排序

> LeetCode - 451 根据字符出现频率排序 
>
> https://leetcode-cn.com/problems/sort-characters-by-frequency/

1、哈希表+vector+sort配合使用

```c++
class Solution {
public:
    string frequencySort(string s) {
        unordered_map<char,int> mp;
        for(char c:s){
            mp[c]++;
        }
        vector<pair<char,int>> vec(mp.begin(),mp.end());
        sort(vec.begin(),vec.end(),[](pair<char,int> a ,pair<char,int> b)
        {
            return a.second > b.second;
        });
        string res;
        for(auto it:vec){
            for(int i=0; i<it.second; ++i){
                res+=it.first;
            }
        }
        return res;
    }
};
```



### 2021.7.11 二分

> LeetCode - 274 H指数 
>
> https://leetcode-cn.com/problems/h-index/

```c++
class Solution {
public:
    int hIndex(vector<int>& citations) {
        //二分查找，确定h
        int left = 0; //h最小是0
        int right = citations.size(); //上限
        while(left<=right){
            int mid = left + (right-left)/2;
            //检验
            int cnt = 0;
            for(auto& num:citations){
                if(num>=mid) cnt++;
            }
            if(cnt>=mid){
                left = mid + 1;
            }else {
                right = mid - 1;
            }
        }
        return right; //注意，这里一定return的是else里面的
    }
};
```



### 2021.7.14 二分&lower_bound()

> LeetCode - 1818 绝对值差和 
>
> https://leetcode-cn.com/problems/minimum-absolute-sum-difference/

```c++
class Solution {
public:
    int minAbsoluteSumDiff(vector<int>& nums1, vector<int>& nums2) {
        int n = nums1.size();
        long long sum = 0;
        for(int i=0; i<n; ++i){
            sum += abs(nums1[i]-nums2[i]); //统计替换前的绝对差值和
        }
        long long ans = sum;
        vector<int> temp(nums1); //复制一份nums1，排序
        sort(temp.begin(),temp.end());
        for(int i=0; i<n; ++i){
            auto it = lower_bound(temp.begin(),temp.end(),nums2[i]);//找到第一个大于等于nums2[i]的下标
            if(it!=temp.end()){
                //找得到比nums2[i]大的数字
                ans = min(ans, sum - abs(nums1[i]-nums2[i]) + abs(*it-nums2[i]));
            }
            if(it!=temp.begin()){
                //两边都要检验
                ans = min(ans, sum - abs(nums1[i]-nums2[i]) + abs(*(--it)-nums2[i]));
            }
        }
        return ans%(int)(1e9+7);
    }
};
```



### 2021.7.15 count&bound&二分

> 剑指Offer 53 - 在排序数组中查找数字Ⅰ
>
> https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/

1、利用count()

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        return count(nums.begin(),nums.end(),target);
    }
};
```

2、利用lower_bound()和upper_bound()

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        return upper_bound(nums.begin(),nums.end(),target) - lower_bound(nums.begin(),nums.end(),target);
    }
};
```

lower_bound( begin,end,num)：从数组的begin位置到end-1位置二分查找第一个**大于或等于**num的数字，找到返回该数字的地址，不存在则返回end。通过返回的地址减去起始地址begin,得到找到数字在数组中的下标。

upper_bound( begin,end,num)：从数组的begin位置到end-1位置二分查找第一个**大于**num的数字，找到返回该数字的地址，不存在则返回end。通过返回的地址减去起始地址begin,得到找到数字在数组中的下标。

3、二分

<img src="https://pic.leetcode-cn.com/b4521d9ba346cad9e382017d1abd1db2304b4521d4f2d839c32d0ecff17a9c0d-Picture1.png" alt="Picture1.png" style="zoom:40%;" />

```c++
class Solution {
public:
    int BinarySearch(vector<int>& nums, int target){
        int left = 0, right = nums.size()-1;
        while(left<=right){
            int mid = left + (right - left)/2;
            if(nums[mid]<=target)left = mid + 1;
            else right = mid - 1;
        }
        return right;
    }

    int search(vector<int>& nums, int target) {
        return BinarySearch(nums,target) - BinarySearch(nums,target-1);
    }
};
```



### 2021.7.17 简单dp(空间优化)

> 剑指 Offer 42. 连续子数组的最大和 
>
> https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n); //dp[i]表示下标到i的连续子数组最大和
        dp[0] = nums[0];
        for(int i=1; i<n; ++i){
            dp[i] = max(dp[i-1]+nums[i],nums[i]);
        }
        return *max_element(dp.begin(),dp.end());
    }
};
```

空间优化: O(n) --> O(1)

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int n = nums.size();
        int maxx = nums[0];
        int res = maxx;
        for(int i=1; i<n; ++i){
            maxx = max(maxx+nums[i],nums[i]);
            res = max(res,maxx);
        }
        return res;
    }
};
```



### 2021.7.18 哈希表&emplace_back

> 面试题 10.02. 变位词组
>
> https://leetcode-cn.com/problems/group-anagrams-lcci/

将词组的字典序作为哈希表的key，将变位词组作为哈希表的value

```c++
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string,vector<string>> mp;
        for(auto& str:strs){
            string key = str;
            sort(key.begin(),key.end());
            mp[key].emplace_back(str);
        }
        vector<vector<string>> res;
        for(auto it=mp.begin(); it!=mp.end(); it++){
            res.emplace_back(it->second);
        }
        return res;
    }
};
```

vector是我们常用的容器，向其中增加元素的常用方法有：emplace_back和push_back两种。

push_back():

首先需要调用构造函数构造一个**临时对象**，然后调用拷贝构造函数将这个临时对象放入容器中，然后释放临时变量。

emplace_back():

这个元素原地构造，**不需要触发拷贝构造和转移构造**。



**C++左值、右值、右值引用详解**  https://blog.csdn.net/hyman_yx/article/details/52044632



### 2021.7.19 排序+滑动窗口

> LeetCode - 1838. 最高频元素的频数 
>
> https://leetcode-cn.com/problems/frequency-of-the-most-frequent-element/

```c++
class Solution {
public:
    int maxFrequency(vector<int>& nums, int k) {
        // 排序
        sort(nums.begin(), nums.end());
        // 定义左右指针
        int i = 0, j = 0, n = nums.size();
        // 统计数值
        long long sum = 0;
        while(j < n) {
            sum += nums[j];
            j++; //右指针右移
            // 如果滑动窗口范围内的值加上所能调整的次数还是无法满足全部相等
            if((long long)nums[j - 1] * (j - i) > sum + k) {
                sum -= nums[i];
                i++; //左指针左移
            }
        }
        return j - i;
    }
};
```



### 2021.7.21 链表&哈希表&双指针

> 剑指 Offer 52. 两个链表的第一公共节点
>
> https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/

1、利用哈希集合

```c++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        unordered_set<ListNode*> st;
        ListNode* temp = headA;
        while(temp){
            st.insert(temp);
            temp = temp->next;
        }
        temp = headB;
        while(temp){
            if(st.count(temp))return temp;
            temp = temp->next;
        }
        return nullptr;
    }
};
```

2、双指针

```c++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if (headA == nullptr || headB == nullptr) {
            return nullptr;
        }
        ListNode *pA = headA, *pB = headB;
        while (pA != pB) {
            pA = pA == nullptr ? headB : pA->next;
            pB = pB == nullptr ? headA : pB->next;
        }
        return pA;
    }
};
```



### 2021.7.25 哈希表+遍历

> LeetCode - 1743 从相邻元素对还原数组 
>
> https://leetcode-cn.com/problems/restore-the-array-from-adjacent-pairs/

```c++
class Solution {
public:
    vector<int> restoreArray(vector<vector<int>>& adjacentPairs) {
        unordered_map<int,vector<int>> mp;
        unordered_set<int> st; //储存已经遍历过的数
        for(int i=0; i<adjacentPairs.size(); ++i){
            mp[adjacentPairs[i][0]].push_back(adjacentPairs[i][1]);
            mp[adjacentPairs[i][1]].push_back(adjacentPairs[i][0]);
        }
        vector<int> ans;
        int begin = 0; //找一个端点
        for(auto& val:mp){
            if(val.second.size()==1){
                begin = val.first;
                st.insert(begin);
                ans.push_back(begin);
                break;
            }
        }

        int temp = begin; //遍历
        while(st.size() < adjacentPairs.size()+1){
            for(int i=0; i<mp[temp].size(); ++i){
                if(st.count(mp[temp][i])) continue;//该数遍历过,直接跳过
                ans.push_back(mp[temp][i]);
                st.insert(mp[temp][i]);
                temp = mp[temp][i];
            }
        }
        return ans;
    }
};
```



### 2021.7.31 二叉树的垂序遍历dfs

> LeetCode - 987 二叉树的垂序遍历
>
> https://leetcode-cn.com/problems/vertical-order-traversal-of-a-binary-tree/

```c++
class Solution {
public:
    void dfs(vector<tuple<int, int, int>>& nodes, TreeNode* node, int row, int col){
        if(!node) return;
        nodes.emplace_back(col,row,node->val);
        dfs(nodes, node->left, row+1, col-1);
        dfs(nodes, node->right, row+1, col+1);
    }
    
    vector<vector<int>> verticalTraversal(TreeNode* root) {
        vector<tuple<int, int, int>> nodes; //[col, row, val]
        dfs(nodes, root, 0, 0);
        sort(nodes.begin(), nodes.end());
        vector<vector<int>> ans;
        int lastcol = INT_MIN;
        for (const auto& [col, row, value]: nodes) {
            if (col != lastcol) {
                lastcol = col;
                ans.emplace_back(); //已经下一列了，开一个新的数组
            }
            ans.back().push_back(value); //不管开没开新数组，都要push进去数据
        }
        return ans;
    }
};
```



### 2021.8.1 优先队列priority_queue

> LeetCode - 1337 矩阵中战斗力最弱的 K 行
>
> https://leetcode-cn.com/problems/the-k-weakest-rows-in-a-matrix/

```c++
class Solution {
public:
    vector<int> kWeakestRows(vector<vector<int>>& mat, int k) {
        priority_queue<pair<int,int>> q; //[power,row]
        //把每一行的战斗力加到优先队列里
        for(int i=0; i<mat.size(); ++i){
            int cnt = 0; //统计这一行的战斗力
            for(int val:mat[i]){
                cnt += val;
            }
            q.push({cnt,i});
        }
        //现在优先队列里面是按战斗力从强到弱排
        //把前面强的pop掉，只剩下需要的k个
        while(q.size()>k) q.pop();
        vector<int> res;
        while(q.size()>0){
            res.push_back(q.top().second);
            q.pop();
        }
        //现在res里是从强到弱排，需要再反转
        reverse(res.begin(),res.end());
        return res;
    }
};
```





### 2021.8.2 Dijkstra算法

> LeetCode - 743 网络延迟时间
>
> https://leetcode-cn.com/problems/network-delay-time/solution/gtalgorithm-dan-yuan-zui-duan-lu-chi-tou-w3zc/
>
> Dijkstra算法讲解视频 https://www.bilibili.com/video/BV1zz4y1m7Nq

Dijkstra算法：

1. 初始化n×n邻接矩阵,初始化为inf/2 (把权重存进去)
2. 初始化visited数组，记录是否被遍历过
3. 初始化dis数组，记录源点到每个点的最短距离，自己到自己是0
4. 遍历n次，找到此时距离源点最短且没有被遍历的点，去更新其他距离，再把visited更新为true

```c++
class Solution {
public:
    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        //防止相加后溢出int边界，所以除以2
        const int inf = INT_MAX/2;

        //邻接矩阵储存边的权重，初始化为max(意为距离无限远)
        vector<vector<int>> mat(n, vector<int>(n, inf));
        //将边与边的权重储存到邻接矩阵中去
        for(auto& val:times){
            int u = val[0] - 1;
            int v = val[1] - 1;
            int cost = val[2];
            mat[u][v] = cost;
        }

        //Dijkstra算法的具体实现
        vector<bool> visited(n, false); //储存该点是否已经遍历过
        vector<int> dis(n, inf); //储存从源节点到每个点的最短距离
        dis[k-1] = 0; //k是源点，自己到自己的距离是0

        for(int i=0; i<n; ++i){
            //找到此时还未确定的，距离源点最短的点
            int cur = -1;
            for(int j=0; j<n; ++j){
                if(!visited[j] && (cur==-1 || dis[j]<dis[cur])){
                    cur = j;
                }
            }

            //已经找到点，从这个点出发去更新距离
            for(int j=0; j<n; ++j){
                dis[j] = min(dis[j], dis[cur]+mat[cur][j]);
            }
            visited[cur] = true; //更新visited
        }

        //找出最长距离(保证所有节点都收到网络信号)
        int ans = *max_element(dis.begin(), dis.end());
        return ans == inf? -1:ans;
    }
};
```



### 2021.8.3 最短无序连续子数组

> LeetCode - 581 最短无序连续子数组
>
> https://leetcode-cn.com/problems/shortest-unsorted-continuous-subarray/

1、把原数组排序好，逐一比较找出左右边界 -- **时间:O(nlogn) 空间:O(n)**

```c++
class Solution {
public:
    int findUnsortedSubarray(vector<int>& nums) {
        vector<int> sortedNums(nums);
        sort(sortedNums.begin(), sortedNums.end());
        if(sortedNums == nums) return 0;
        
        int left = 0;
        int right = nums.size() - 1;
        while(nums[left] == sortedNums[left]) left++;
        while(nums[right] == sortedNums[right]) right--;
        return right - left + 1;
    }
};
```

2、一次遍历 -- **时间:O(n) 空间:O(1)**

```c++
class Solution {
public:
    int findUnsortedSubarray(vector<int>& nums) {
        int n = nums.size();
        int maxn = INT_MIN, right = -1;
        int minn = INT_MAX, left = -1;
        for (int i = 0; i < n; i++) {
            if (nums[i] >= maxn) {
                maxn = nums[i];
            } else {
                right = i;
            }
            if (nums[n - i - 1] <= minn) {
                minn = nums[n - i - 1];
            } else {
                left = n - i - 1;
            }
        }
        return right == -1 ? 0 : right - left + 1;
    }
};
```





### 2021.8.4 二分查找

> LeetCode - 611 有效三角形的个数
>
> https://leetcode-cn.com/problems/valid-triangle-number/

```c++
class Solution {
public:
    int triangleNumber(vector<int>& nums) {
        int n = nums.size();
        if(n < 3) return 0;
        sort(nums.begin(), nums.end());
        int cnt = 0;
        for(int i=0; i<n; ++i){
            for(int j=i+1; j<n; ++j){
                //二分查找 找到能和nums[i],nums[j]组成三角形的最大下标k,则j~k中间的也都可以组成三角形
                int left = j+1, right = n-1;
                int k = j; //记录最大下标k
                while(left <= right){
                    int mid = left + (right - left) / 2;
                    if(nums[i] + nums[j] > nums[mid]){
                        //可以组成三角形，记录当前位置，并继续向右搜索
                        k = mid;
                        left = mid + 1;
                    }else{
                        //不能组成三角形，当前边太大，向左搜索
                        right = mid - 1;
                    }
                }
                cnt += k-j;
            }
        }
        return cnt;
    }
};
```



### 2021.8.5 拓扑排序 

> LeetCode - 802. 找到最终的安全状态
>
> https://leetcode-cn.com/problems/find-eventual-safe-states/solution/gtalgorithm-san-ju-hua-jiao-ni-wan-zhuan-xf5o/

拓扑排序：

1. 把所有入度为0的点加入队列
2. pop队首，更新邻接的点(入度-1)，再检查更新后是否有点入度为0，加到队列里
3. 直到队列为空

```c++
class Solution {
public:
    vector<int> eventualSafeNodes(vector<vector<int>>& graph) {
        //拓扑排序是找到图中入度为 0 的节点，以及仅由入度为 0 节点所指向的节点 
        //而本题是找到图中出度为 0 的节点，以及仅指向出度为 0 节点的节点
        //所以先要把原图转化为反图(箭头反转)
        int n = graph.size();
        vector<vector<int>> r_graph(n); //反图，将原图的箭头反转
        vector<int> inDeg(n, 0); //记录反图中每个点的入度(即原图的出度)
        for(int i = 0; i < n ; ++i){
            for(int x : graph[i]){
                r_graph[x].push_back(i);
            }
            inDeg[i] = graph[i].size(); //反图入度 = 原图出度
        }
        //拓扑排序
        queue<int> q;
        //首先将所有入度为0的点加入队列
        for(int i = 0; i < n ; ++i){
            if(inDeg[i]==0) q.push(i);
        }

        while(!q.empty()){
            int cur = q.front();
            q.pop();

            for(int x : r_graph[cur]){
                //将队列首点的邻接点入度-1
                inDeg[x]--;
                //如果更新完之后的入度变为0，则加入队列
                if(inDeg[x] == 0) q.push(x);
            }
        }

        //最终入度(原图出度)为 0 的点为安全点
        vector<int> res;
        for(int i = 0; i < n ; ++i){
            if(inDeg[i]==0) res.push_back(i);
        }
        sort(res.begin(), res.end());
        return res;
    }
};
```



### 2021.8.6 状态压缩BFS

> LeetCode - 847. 访问所有节点的最短路径
>
> https://leetcode-cn.com/problems/shortest-path-visiting-all-nodes/
>
> 详细题解：https://leetcode-cn.com/problems/shortest-path-visiting-all-nodes/solution/gtalgorithm-tu-jie-fa-ba-hardbian-cheng-v5knb/

```c++
class Solution {
public:
    int shortestPathLength(vector<vector<int>>& graph) {
        int n = graph.size();

        // 1.初始化队列及标记数组，存入起点
        queue< tuple<int, int, int> > q; // 三个属性分别为 idx, mask, dist
        vector<vector<bool>> vis(n, vector<bool>(1 << n)); // 节点编号及当前状态
        for(int i = 0; i < n; i++) {
            q.push({i, 1 << i, 0}); // 存入起点，起始距离0，标记
            vis[i][1 << i] = true;
        }

        // 开始搜索
        while(!q.empty()) {
            auto [cur, mask, dist] = q.front(); // 弹出队头元素
            q.pop();

            // 找到答案，返回结果
            if(mask == (1 << n) - 1) return dist;

            // 扩展
            for(int x : graph[cur]) {
                int nextmask = mask | (1 << x);
                if(!vis[x][nextmask]) {
                    q.push({x, nextmask, dist + 1});
                    vis[x][nextmask] = true;
                }
            }
        }
        return 0;
    }
};
```



### 2021.8.7 拓扑排序的应用

> LeetCode - 457. 环形数组是否存在循环
>
> https://leetcode-cn.com/problems/circular-array-loop/

1、拓扑排序，一个图中是否有环可以用拓扑排序来检验

```c++
class Solution {
public:
    vector<vector<int>> graph; // 邻接表，将原数组转化为图
    vector<int> inDeg;         // 储存每个点的入度 

    // 拓扑排序，检验图中是否有环
    bool TopSort(int n){
        queue<int> q;
        // 首先把入度为0的节点加入队列
        for(int i = 0; i < n; ++i){
            if(!inDeg[i]) q.push(i);
        }

        while(!q.empty()){
            // 弹出队头元素
            int cur = q.front();
            q.pop();

            // 找到以cur为起点的有向边，终点入度-1   
            for(int x : graph[cur]){
                inDeg[x]--;
                // 若更新后入度也为0，则入队
                if(!inDeg[x]) q.push(x);
            }
        }

        // 检验拓扑排序后所有点的入度
        for(int i = 0; i < n; ++i){
            // 仍有入度不为0的点，说明存在环
            if(inDeg[i]) return true;
        }
        return false;
    }

    bool circularArrayLoop(vector<int>& nums) {
        int n = nums.size();
        graph.resize(n);
        inDeg.resize(n);

        // 先处理正向边(nums[i] > 0)的情况
        for(int i = 0; i < n; ++i){
            int next = ((i + nums[i]) % n + n) % n;
            if( nums[i] <= 0 || next == i) continue; // 忽略负向边和自环的情况
            graph[i].push_back(next);
            inDeg[next]++;
        }

        if(TopSort(n)) return true; // 检验正向边

        // 清空数组
        graph.clear();
        graph.resize(n);
        inDeg.clear();
        inDeg.resize(n);

        // 再处理负向边(nums[i] < 0)的情况
        for(int i = 0; i < n; ++i){
            int next = ((i + nums[i]) % n + n) % n;
            if( nums[i] > 0 || next == i) continue; // 忽略正向边和自环的情况
            graph[i].push_back(next);
            inDeg[next]++;
        }

        if(TopSort(n)) return true; // 检验负向边

        return false;
    }
};
```

2、哈希集合

```c++
class Solution {
public:
    bool circularArrayLoop(vector<int>& nums) {
        int n = nums.size();
        // 遍历nums, 遍历到i说明从nums[i]出发，检验从该点出发是否有环
        for(int i = 0; i < n; ++i){
            unordered_set<int> st; // set储存已遍历的index
            int cur = i;
            st.insert(cur);
            while(1){
                int next = ( (cur + nums[cur] % n) + n ) % n;
                if(nums[cur] * nums[next] <= 0 || next == cur) break; // 不同方向和自环的情况不符合要求
                if(st.count(next)) return true;
                st.insert(next);
                cur = next; // 迭代
            }
        }
        return false;
    }
};
```

3、快慢指针

```c++
class Solution {
public:
    bool circularArrayLoop(vector<int>& nums) {
        int n = nums.size();
        auto next = [&](int x){
            return ((x + nums[x]) % n + n) %n;
        };
        // 遍历nums,检验从i出发是否有环
        for(int i = 0; i < n; ++i){
            if(!nums[i]) continue; // 0肯定不存在环
            int slow = i, fast = next(i);
            while(nums[slow] * nums[fast] > 0 && nums[slow] * nums[next(fast)] > 0){
                if(slow == fast){
                    // fast把slow套圈了
                    if(slow != next(slow)) return true; // 不是自环，符合要求
                    if(slow == next(slow)) break; // 排除自环的情况
                }
                slow = next(slow);
                fast = next(next(fast));
            }
        }
        return false;
    }
};
```

附：匿名函数

https://blog.csdn.net/zhang14916/article/details/101058089





### 2021.8.8 滚动数组

> LeetCode - 1137. 第N个泰波那契数
>
> https://leetcode-cn.com/problems/n-th-tribonacci-number/

```c++
class Solution {
public:
    int tribonacci(int n) {
        if(n == 0) return 0;
        if(n == 1) return 1;
        if(n == 2) return 1;

        int p = 0, q = 0, r = 1, s = 1;
        for(int i = 3; i <= n; ++i){
            p = q;
            q = r;
            r = s;
            s = p + q + r;
        }
        return s;
    }
};
```



### 2021.8.9 优先队列&多指针

> LeetCode - 313 超级丑数
>
> https://leetcode-cn.com/problems/super-ugly-number/

1、优先队列法

```c++
class Solution {
public:
    int nthSuperUglyNumber(int n, vector<int>& primes) {
        unordered_set<long> seen; //哈希集合储存遍历过的数
        priority_queue<long, vector<long>, greater<long>> q;
        seen.insert(1);
        q.push(1);
        int ugly = 0;
        for(int i = 0; i < n; ++i){
            long cur = q.top();
            q.pop();
            ugly = (int)cur;
            for(int x : primes){
                long next = cur * x;
                if(!seen.count(next)){
                    seen.insert(next);
                    q.push(next);
                }
            }
        }
        return ugly;
    }
};
```

2、多指针法

```c++
class Solution {
public:
    int nthSuperUglyNumber(int n, vector<int>& primes) {
        vector<int> dp(n + 1); //dp[i]代表第i个超级丑数
        dp[1] = 1;
        int m = primes.size();
        vector<int> pointer(m, 1);
        for(int i = 2; i <= n; ++i){
            vector<int> nums(m);
            for(int j = 0; j < m; ++j){
                nums[j] = primes[j] * dp[pointer[j]];
            }
            int minNum = *min_element(nums.begin(), nums.end());
            dp[i] = minNum;
            for(int j = 0; j < m; ++j){
                if(nums[j] == minNum){
                    pointer[j]++;
                }
            }
        }
        return dp[n];
    }
};
```



### 2021.8.16 回溯

> LeetCode - 526. 优美的排列
>
> https://leetcode-cn.com/problems/beautiful-arrangement/

```c++
class Solution {
public:
    vector<bool> vis; // 记录是否被遍历过
    int num = 0;

    void backtrack(int n, int index){
        if(index == n + 1){
            num++; // 找到一个符合要求的序列, num+1
            return;
        }
        for(int i = 1; i < n + 1; ++i){
            // 枚举第index个符合要求且没有被遍历过的元素
            if(!vis[i] && (i % index == 0 || index % i == 0)){
                vis[i] = true; // 将该元素标记为已遍历
                backtrack(n, index + 1); // 向后继续寻找第index + 1 个满足要求的元素
                vis[i] = false; // 回溯，标记为没有遍历过，后面还能继续用
            }
        }
    }

    int countArrangement(int n) {
        vis.resize(n + 1);
        backtrack(n, 1);
        return num;
    }
};
```

> LeetCode - 46. 全排列
>
> https://leetcode-cn.com/problems/permutations/

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    vector<bool> vis;

    void backtrack(vector<int>& nums, vector<int>& path){
        int n = nums.size();
        if(path.size() == n){
            res.push_back(path);
            return;
        }
        for(int i = 0; i < n; ++i){
            if(vis[i]) continue;
            path.push_back(nums[i]);
            vis[i] = true;
            backtrack(nums, path);
            path.pop_back();
            vis[i] = false;
        }
    } 

    vector<vector<int>> permute(vector<int>& nums) {
        int n = nums.size();
        vis.resize(n);
        backtrack(nums, path);
        return res;
    }
};
```

升级版解法：

```c++
class Solution {
public:
    vector<vector<int>> res;

    void backtrack(vector<int>& nums, int index){
        int n = nums.size();
        if(index == nums.size() - 1){
            res.push_back(nums);
            return;
        }

        for(int i = index; i < n; ++i){
            swap(nums[i], nums[index]);
            backtrack(nums, index + 1);
            swap(nums[index], nums[i]);
        }
    } 

    vector<vector<int>> permute(vector<int>& nums) {
        backtrack(nums, 0);
        return res;
    }
};
```



### 2021.8.19 双指针-反转元音字母

> LeetCode - 345. 反转字符串中的元音字母
>
> https://leetcode-cn.com/problems/reverse-vowels-of-a-string/

```c++
class Solution {
public:
    bool isVowel(char c){
        string vowel = "aeiouAEIOU";
        if(vowel.find(c) != -1) return true;
        return false;
    }

    string reverseVowels(string s) {
        // 双指针法
        int n = s.size();
        int i = 0, j = n - 1;
        while(i < j){
            while(i < n && !isVowel(s[i])){
                i++;
            }
            while(j > 0 && !isVowel(s[j])){
                j--;
            }
            if(i < j){
                swap(s[i++], s[j--]);
            }
        }
        return s; 
    }
};
```



### 2021.8.30 前缀和&按权重随机

> LeetCode - 528. 按权重随机选择
>
> https://leetcode-cn.com/problems/random-pick-with-weight/

```c++
class Solution {
public:
    vector<int> preSum;
    Solution(vector<int>& w) {
        preSum.push_back(w[0]);
        for(int i = 1; i < w.size(); ++i){
            preSum.push_back(preSum.back() + w[i]);
        }
    }
    
    int pickIndex() {
        int random = rand() % preSum.back();
        return  upper_bound(preSum.begin(), preSum.end(), random) - preSum.begin();
    }
};
```



### 2021.9.5 用Rand7() 实现Rand10()

> LeetCode - 470. 用Rand7() 实现Rand10()
>
> https://leetcode-cn.com/problems/implement-rand10-using-rand7/solution/cong-zui-ji-chu-de-jiang-qi-ru-he-zuo-dao-jun-yun-/

文章基于这样一个事实 (randX() - 1)*Y + randY() 可以等概率的生成[1, X * Y]范围的随机数

```c++
// The rand7() API is already defined for you.
// int rand7();
// @return a random integer in the range 1 to 7

class Solution {
public:
    int rand10() {
        int a = rand7();
        int b = rand7();
        int num = (a - 1) * 7 + b;  // rand49()
        if(num <= 40) return num % 10 + 1;

        a = num - 40; // rand9()
        b = rand7();
        num = (a - 1) * 7 + b;  // rand63()
        if(num <= 60) return num % 10 + 1;

        a = num - 60; // rand3()
        b = rand7();
        num = (a - 1) * 7 + b;  // rand21()
        if(num <= 20) return num % 10 + 1;

        return 1;
    }
};
```



### 2021.9.12 有效的括号字符串

> LeetCode - 678. 有效的括号字符串
>
> https://leetcode-cn.com/problems/valid-parenthesis-string/

1、栈

```c++
class Solution {
public:
    bool checkValidString(string s) {
        int n = s.size();
        if(!n) return true;
        stack<int> left;
        stack<int> star;
        for(int i = 0; i < n; ++i){
            if(s[i] == '('){
                left.push(i);
            }else if(s[i] == '*'){
                star.push(i);
            }else if(s[i] == ')'){
                //优先使用左括号，若没有再使用星号
                if(!left.empty()){
                    left.pop();
                }else if(!star.empty()){
                    star.pop();
                }else{
                    return false;
                }
            }
        }
        while(!left.empty() && !star.empty()){
            int leftIndex = left.top();
            left.pop();
            int starIndex = star.top();
            star.pop();
            if(leftIndex > starIndex) return false; //左括号必须在星号的左边
        }
        return left.empty();
    }
};
```

2、贪心

```c++
class Solution {
public:
    bool checkValidString(string s) {
        int n = s.size();
        if(!n) return true;
        //未匹配的左括号的最大值和最小值
        //min代表*用作右括号或者空
        //max代表*用作了左括号
        int minCnt = 0, maxCnt = 0;
        for(auto c:s){
            if(c == '('){
                ++minCnt;
                ++maxCnt;
            }else if(c == ')'){
                minCnt = max(minCnt - 1, 0);
                --maxCnt;
                if(maxCnt < 0){
                    //没有与这个右括号相对应的左括号
                    return false;
                }
            }else if(c == '*'){
                ++maxCnt;
                minCnt = max(minCnt - 1, 0);
            }
        }
        return minCnt == 0;
    }
};
```





### 2021.9.16 2/3/4数之和

> 1、两数之和 
>
> https://leetcode-cn.com/problems/two-sum/

哈希表

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int n = nums.size();
        unordered_map<int, int> mp;
        for(int i = 0; i < n; ++i){
            auto it = mp.find(target - nums[i]);
            if(it != mp.end()){
                return {i, mp[target - nums[i]]};
            }
            mp[nums[i]] = i;
        }
        return {};
    }
};
```

> 2、三数之和
>
> https://leetcode-cn.com/problems/3sum/

排序+双指针

```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        vector<vector<int>> ans;
        for(int first = 0 ; first < n; ++first){
            //略过重复情况
            if(first > 0 && nums[first] == nums[first - 1]){
                continue;
            }
            //第三个数从后往前搜索
            //不需要对于每个b，c都从最后面开始搜索，因为b变大，c必须变小
            int third = n - 1; 
            int target = -nums[first];
            for(int second = first + 1; second < n; ++second){
                //略过重复情况
                if(second > first + 1 && nums[second] == nums[second - 1]){
                    continue;
                }
                while(second < third && nums[second] + nums[third] > target){
                    third--;
                }
                //双指针重合，若b再往前走，b>c，无符合要求的答案
                if(second == third){
                    break;
                } 
                if(nums[second] + nums[third] == target){
                    ans.push_back({nums[first], nums[second], nums[third]});
                }
            }
        }
        return ans;
    }
};
```

```c++
// while写法
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        vector<vector<int>> ans;
        for(int first = 0 ; first < n; ++first){
            //略过重复情况
            if(first > 0 && nums[first] == nums[first - 1]){
                continue;
            }
            int left = first + 1;
            int right = n - 1;
            while(left < right){
                if(nums[first] + nums[left] + nums[right] > 0){
                    right--;
                }else if(nums[first] + nums[left] + nums[right] < 0){
                    left++;
                }else if(nums[first] + nums[left] + nums[right] == 0){
                    ans.push_back({nums[first], nums[left], nums[right]});
                    //找到一个答案，去重
                    while(right > left && nums[right] == nums[right - 1]){
                        right--;
                    }
                    while(right > left && nums[left] == nums[left + 1]){
                        left++;
                    }
                    right--;
                    left++;
                }
            }    
        }
        return ans;
    }
};
```



> 3、四数之和
>
> https://leetcode-cn.com/problems/4sum/

与三数之和思路类似

```c++
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        vector<vector<int>> res;
        for(int k = 0; k < n; ++k){
            if(k > 0 && nums[k] == nums[k - 1]){
                continue;
            }
            for(int i = k + 1; i < n; ++i){
                if(i > k + 1 && nums[i] == nums[i - 1]){
                    continue;
                }
                int left = i + 1;
                int right = n - 1;
                int target1 = target - nums[k] - nums[i];
                while(left < right){
                    if(nums[left] + nums[right] > target1){
                        right--;
                    }else if(nums[left] + nums[right] < target1){
                        left++;
                    }else if(nums[left] + nums[right] == target1){
                        res.push_back({nums[k], nums[i], nums[left], nums[right]});
                        //找到一个答案，去重
                        while(right > left && nums[right] == nums[right - 1]){
                            right--;
                        }
                         while(right > left && nums[left] == nums[left + 1]){
                            left++;
                        }
                        right--;
                        left++;
                    }
                }
            }
        }
        return res;
    }
};
```

 



### 2021.9.20 最长递增子序列

> LeetCode 300. 最长递增子序列
>
> https://leetcode-cn.com/problems/longest-increasing-subsequence/

1、dp

```c++
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n); // dp[i] 表示 以nums[i]为结尾的最长递增子序列
        for(int i = 0; i < n; ++i){
            dp[i] = 1; // 自身必须被选取
            for(int j = 0; j < i; ++j){
                if(nums[j] < nums[i]){
                    dp[i] = max(dp[i], dp[j] + 1);
                }
            }
        }
        return *max_element(dp.begin(), dp.end());
    }
};
```



### 2021.9.22 分隔链表

> LeetCode - 725. 分隔链表
>
> https://leetcode-cn.com/problems/split-linked-list-in-parts/

```c++
class Solution {
public:
    vector<ListNode*> splitListToParts(ListNode* head, int k) {
        // 遍历链表，得到链表长度
        ListNode* temp = head;
        int n = 0;
        while(temp){
            temp = temp->next;
            ++n;
        }
        int quotient = n / k; // 每份的个数
        int reminder = n % k; // 多出来的几个，放在前面，一组一个

        vector<ListNode*> res(k, nullptr);
        ListNode* curr = head;
        for(int i = 0; i < k && curr; ++i){
            res[i] = curr;
            int thisSize = quotient;
            if(i < reminder) ++thisSize; // 余数还没用完，加到这一组里
            // 找到这一组的尾结点，断开
            for(int j = 0; j < thisSize - 1; ++j){
                curr = curr->next;
            }
            //断开这个节点与下一节点的联系
            ListNode* tempNext = curr->next;
            curr->next = nullptr;
            curr = tempNext;
        }
        return res;
    }
};
```



### 2021.9.25 最长公共子序列

> LeetCode - 1143. 最长公共子序列
>
> https://leetcode-cn.com/problems/longest-common-subsequence/

动态规划：

```c++
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int n = text1.size();
        int m = text2.size();
        // dp[i][j] 代表 text1长度为i的前缀 和 text2长度为j的前缀 的最长公共子序列
        // 初始条件为 dp[i][0] = dp[0][j] = 0;
        vector<vector<int>> dp(n + 1, vector<int>(m + 1));
        for(int i = 1; i <= n; ++i){
            for(int j = 1; j <= m; ++j){
                if(text1[i - 1] == text2[j - 1]){
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }else{
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[n][m]; 
    }
};
```



升级版：

> LeetCode 583. 两个字符串的删除操作
>
> https://leetcode-cn.com/problems/delete-operation-for-two-strings/

用上面这题的动态规划：

```c++
class Solution {
public:
    int minDistance(string word1, string word2) {
        int n = word1.size();
        int m = word2.size();
        // dp[i][j] 代表 text1长度为i的前缀 和 text2长度为j的前缀 的最长公共子序列
        // 初始条件为 dp[i][0] = dp[0][j] = 0;
        vector<vector<int>> dp(n + 1, vector<int>(m + 1));
        for(int i = 1; i <= n; ++i){
            for(int j = 1; j <= m; ++j){
                if(word1[i - 1] == word2[j - 1]){
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }else{
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return m + n - 2 * dp[n][m]; 
    }
};
```

上面是间接的

下面用直接的动态规划：

```c++
class Solution {
public:
    int minDistance(string word1, string word2) {
        int m = word1.size();
        int n = word2.size();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1));

        for (int i = 1; i <= m; ++i) {
            dp[i][0] = i;
        }
        for (int j = 1; j <= n; ++j) {
            dp[0][j] = j;
        }
        for (int i = 1; i <= m; i++) {
            char c1 = word1[i - 1];
            for (int j = 1; j <= n; j++) {
                char c2 = word2[j - 1];
                if (c1 == c2) {
                    // 两个字符一样，不需要多加删除的步数
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    // 两个字符不一样，需要多加一步
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + 1;
                }
            }
        }

        return dp[m][n];
    }
};
```



### 2021.9.26 网格游戏

> LeetCode - 5882. 网格游戏
>
> https://leetcode-cn.com/problems/grid-game/

```c++
class Solution {
public:
    long long gridGame(vector<vector<int>>& grid) {
        int n = grid[0].size();
        vector<vector<long long>> preSum(2, vector<long long>(n)); // 计算每一行的前缀和
        preSum[0][0] = grid[0][0];
        preSum[1][0] = grid[1][0];
        for(int i = 0; i < 2; ++i){
            for(int j = 1; j < n; ++j){
                preSum[i][j] = preSum[i][j - 1] + grid[i][j];
            }
        }

        // 初始值的情况为:bot1直接从下标0就往下走
        long long ans = preSum[0][n - 1] - preSum[0][0];
        // 遍历:从下标1开始拐~下标n-1开始拐
        for(int j = 1; j < n; ++j){
            // 括号里取max是因为bot2要最大化自己的点数
            // 括号外取min是因为bot1要最小化bot2的点数
            // preSum[0][n - 1] - preSum[0][j]是第一行剩下的点数
            // preSum[1][j - 1]是第二行剩下的点数
            ans = min(ans, max(preSum[0][n - 1] - preSum[0][j], preSum[1][j - 1]));
        }
        return ans;
    }
};
```





### 2021.9.28 二叉树&dfs&前缀和

> LeetCode - 437. 路径总和Ⅲ
>
> https://leetcode-cn.com/problems/path-sum-iii/

**1、dfs**(双递归，效率不好) 第二种前缀和法要更好一点

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int dfs(TreeNode* root, int targetSum){
        // 空节点直接返回 rootSum = 0
        if(!root){
            return 0;
        }
		
        // 记录从该节点出发，有多少种符合要求的路径
        int ret = 0;
        if(root->val == targetSum){
            ret++;
        }
		
        // dfs
        ret += dfs(root->left, targetSum - root->val);
        ret += dfs(root->right, targetSum - root->val);
        return ret;
    }


    int pathSum(TreeNode* root, int targetSum) {
        if(!root) return 0;
        int ret = dfs(root, targetSum);  		// 初始化为从根节点出发的路径数
        ret += pathSum(root->left, targetSum);  // 递归调用pathSum，dfs计算总路径数
        ret += pathSum(root->right, targetSum);
        return ret;
    }
};
```

Time Complexity: O(N<sup>2</sup>) 有很多重复计算(类似斐波那契数列的递归)

Space Complexity: O(N) 递归调用需要在栈上开辟空间



**2、前缀和**

```c++
class Solution {
public:
    unordered_map<int, int> preSum; // [preSum, count]

    int dfs(TreeNode* root, long long curr, int targetSum){
        if(!root){
            return 0;
        }

        int ret = 0;	   // 记录从顶点到当前节点，一共有多少条路径
        curr += root->val; // 更新当前前缀和curr
        // 当前前缀和为curr
        // 如果之前路径有前缀和为curr- targetSum
        // 则curr - (curr - targerSum) = targetSum
        if(preSum.count(curr - targetSum)){
            ret += preSum[curr - targetSum];
        }

        preSum[curr]++; // 更新preSum
        ret += dfs(root->left, curr, targetSum);
        ret += dfs(root->right, curr, targetSum);
        preSum[curr]--; // 回溯

        return ret;
    }


    int pathSum(TreeNode* root, int targetSum) {
        preSum[0] = 1; // 前缀和为0的路径初始化为1(空路径)
        return dfs(root, 0, targetSum);
    }
};
```

Time Complexity: O(N) 前缀和只需要遍历一次二叉树

Space Complexity: O(N)



### 2021.9.30 矩形面积

> LeetCode - 223. 矩形面积
>
> https://leetcode-cn.com/problems/rectangle-area/

```c++
class Solution {
public:
    int computeArea(int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2) {
        int area1 = (ax2 - ax1) * (ay2 - ay1);
        int area2 = (bx2 - bx1) * (by2 - by1);
        int overlapX = max(min(ax2, bx2) - max(ax1, bx1), 0);
        int overlapY = max(min(ay2, by2) - max(ay1, by1), 0);
        return area1 + area2 - overlapX * overlapY;
    }
};
```



### 2021.10.2 数字转十六进制

> LeetCode - 405. 数字转换十六进制
>
> https://leetcode-cn.com/problems/convert-a-number-to-hexadecimal/

```c++
class Solution {
public:
    string toHex(int num) {
        string res;
        long N = num; 
        if (N == 0) return "0";
        string dict = "0123456789abcdef";
        if (N < 0) N = N + 0x100000000; 
        while (N > 0)
        {
            long lastDigit = N % 16;
            N /= 16;
            res = dict[lastDigit] + res;
        }
        return res;
    }
};
```



### 2021.10.6 第三大的数

1、排序

```c++
class Solution {
public:
    int thirdMax(vector<int>& nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end()); // 从小到大排序
        int cnt = 1; // 记录现在是第几大的数
        for(int i = n - 2; i >= 0; --i){
            if(nums[i] != nums[i + 1]){
                ++cnt;
                if(cnt == 3) return nums[i];
            }
        }
        return nums[n - 1];
    }
};
```

2、有序集合set

```c++
class Solution {
public:
    int thirdMax(vector<int> &nums) {
        set<int> s;
        for (int num : nums) {
            s.insert(num);
            if (s.size() > 3) {
                s.erase(s.begin());
            }
        }
        // 如果遍历完数组，集合大小为3，就返回最小的数(数组中第三大的数)
        // 如果集合大小小于3，就返回最大的数(数组中最大的数)
        return s.size() == 3 ? *s.begin() : *s.rbegin();
    }
};
```

3、一次遍历

用三个变量a/b/c 维护前三大的数

```c++
class Solution {
public:
    int thirdMax(vector<int>& nums) {
        int n = nums.size();
        long a = LONG_MIN, b = LONG_MIN, c = LONG_MIN;
        for(int i = 0; i < n; ++i){
            if(nums[i] > a){
                c = b;
                b = a;
                a = nums[i];
            }else if(nums[i] > b && nums[i] < a){
                c = b;
                b = nums[i];
            }else if(nums[i] > c && nums[i] < b){
                c = nums[i];
            }
        }
        return c == LONG_MIN ? a : c;
    }
};
```





### 2021.10.7 找单词数

> LeetCode - 434. 字符串中的单词数
>
> https://leetcode-cn.com/problems/number-of-segments-in-a-string/

```c++
class Solution {
public:
    int countSegments(string s) {
        int res = 0;
        // 记录当前状态
        // t=0表示空格状态
        // t=1表示识别到字符的状态
        int t = 0; 
        for(char c:s){
            if(c != ' ' && t == 0){
                // 识别到新单词并且此时是空格状态
                ++res;
                t = 1;
            }else if(c == ' '){
                //识别到空格，进入空格状态
                t = 0;
            }
        }
        return res;
    }
};
```



### 2021.10.9 两数相加

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode* head = nullptr, *tail = nullptr;
        int next = 0;
        while(l1 || l2){
            int num1 = l1 == nullptr ? 0 : l1->val;
            int num2 = l2 == nullptr ? 0 : l2->val;
            int temp = num1 + num2 + next;
            if(!head){
                head = tail = new ListNode(temp % 10);
            }else{
                tail->next = new ListNode(temp % 10);
                tail = tail->next;
            }
            next = temp / 10;
            if(l1){
                l1 = l1->next;
            }
            if(l2){
                l2 = l2->next;
            }
        }
        // 最后一位加完还有进位情况
        if(next){
            tail->next = new ListNode(next);
            tail = tail->next;
        }
        return head;
    }
};
```



### 2021.10.11 堆排序

堆排序：

从下向上初始化建堆(i = n / 2 -1; i  >= 0)

-->开始排序:倒序(i = n - 1; i > 0)取顶堆(交换v[0]和v[i])

-->重新构建堆v[0]~v[i-1]



作用: topK, 优先队列

```C++
/*
* 调整堆
* i:父节点下标
* n:调整到哪里为止(包含下标n)
*/
void HeapAdjust(vector<int>& v, int i, int n);

// 5、堆排序 O(nlogn)
void HeapSort(vector<int>& v)
{
	int n = v.size();

	// 初始化构建大顶堆
	for (int i = n / 2 - 1; i >= 0; --i)
	{	
		// 从下向上构建大顶堆
		HeapAdjust(v, i, n - 1);

		//cout << "初始化大顶堆:";
		//for (int i : v) {
		//	cout << i << " ";
		//}
		//cout << endl;
	}

	for (int i = n - 1; i > 0; --i)
	{	
		swap(v[i], v[0]); // v[0]是大顶堆的顶，目前最大值，放到最后面
		HeapAdjust(v, 0, i - 1); // 将v[0]-v[i - 1]重新构造大顶堆

		//cout << "进行排序:";
		//for (int i : v) {
		//	cout << i << " ";
		//}
		//cout << endl;
	}
}

// 调整堆
void HeapAdjust(vector<int>& v, int i, int n)
{
	int parent = i; // 记录此父节点
	for (int child = 2 * i + 1; child <= n; child = 2 * child + 1)
	{
		if (child < n && v[child] < v[child + 1])
		{
			// child不是最后一个元素
			// 并且左子节点小于右子节点，就选择右边
			++child;
		}

		if (v[parent] >= v[child])
		{	
			// 父亲比儿子及其下面的所有都要大，直接退出循环
			break;
		}
		else if (v[parent] < v[child])
		{
			swap(v[parent], v[child]); // 将父节点与大的子节点交换
			parent = child;	// 继续向下调整
		}
	}
}
```



### 2021.10.13 双栈实现队列

https://leetcode-cn.com/problems/implement-queue-using-stacks/

```c++
class MyQueue {
public:
    stack<int> stIn;   // 输入栈
    stack<int> stOut;  // 输出栈

    MyQueue() {

    }
    
    void push(int x) {
        stIn.push(x);
    }
    
    int pop() {
        // 如果输出栈为空 就把输入栈所有元素压入输出栈
        if(stOut.empty()){
            while(!stIn.empty()){
                stOut.push(stIn.top());
                stIn.pop();
            }
        }
        int res = stOut.top();
        stOut.pop();
        return res;
    }
    
    int peek() {
        int temp = this->pop(); // 复用pop()
        stOut.push(temp);		// pop弹出了第一个元素，压回去
        return temp;
    }
    
    bool empty() {
        return stIn.empty() && stOut.empty();
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue* obj = new MyQueue();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->peek();
 * bool param_4 = obj->empty();
 */
```

