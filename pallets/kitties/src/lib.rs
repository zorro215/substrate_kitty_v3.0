#![cfg_attr(not(feature = "std"), no_std)]

use codec::{Decode, Encode};
use frame_support::{dispatch::DispatchResult, ensure, RuntimeDebug, traits::{Currency, ExistenceRequirement, Get, Randomness},
};
#[cfg(feature = "std")]
use frame_support::traits::GenesisBuild;
use frame_system::offchain::{SendTransactionTypes, SubmitTransaction};
use orml_nft::Module as NftModule;
use orml_utilities::with_transaction_result;
use pallet_timestamp as timestamp;

#[cfg(feature = "std")]
use serde::{Deserialize, Serialize};
use sp_io::hashing::blake2_128;
use sp_runtime::{
    offchain::storage_lock::{BlockAndTime, StorageLock},
    RandomNumberGenerator,
    traits::BlakeTwo256, transaction_validity::{
        InvalidTransaction, TransactionSource, TransactionValidity, ValidTransaction,
    },
};
use sp_std::{convert::TryInto, vec::Vec};
pub use pallet::*;
pub use weights::WeightInfo;

#[cfg(test)]
mod tests;
#[cfg(feature = "runtime-benchmarks")]
mod benchmarking;
pub mod weights;

//**结构体定义**
/// Kitty对象作为NFT的元素
#[derive(Encode, Decode, Clone, RuntimeDebug, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
pub struct Kitty(pub [u8; 16]);

/// Kitty 公共方法定义
impl Kitty {
    pub fn gender(&self) -> KittyGender {
        if self.0[0] % 2 == 0 {
            KittyGender::Male
        } else {
            KittyGender::Female
        }
    }
}

#[derive(Encode, Decode, Clone, RuntimeDebug, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
/// 性别枚举
pub enum KittyGender {
    Male,
    Female,
}

/// 实现 Default 默认值trait
impl Default for KittyGender {
    fn default() -> Self {
        KittyGender::Male
    }
}

#[derive(Encode, Decode, Clone, RuntimeDebug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
pub struct KittyDto<TokenId, Balance, Moment> {
    id: TokenId,
    dna: [u8; 16],
    birthday: Moment,
    sale_status: bool,
    price: Balance,
    sex: KittyGender,
    // 后面再实现家庭关系
    // father: [u8; 16],
    // mather: [u8; 16],
    // children: BTreeSet<[u8; 16]>,
}


//**泛型类型关联**
/// NFT-TokenId
type KittyIndexOf<T> = <T as orml_nft::Config>::TokenId;
/// Kitty 创建时间 (birthday)
type MomentOf<T> = <T as pallet_timestamp::Config>::Moment;
/// Kitty 售价
type BalanceOf<T> = <<T as Config>::Currency as Currency<<T as frame_system::Config>::AccountId>>::Balance;


#[frame_support::pallet]
pub mod pallet {
    use frame_support::pallet_prelude::*;
    use frame_system::pallet_prelude::*;

    use super::*;
    use sp_runtime::sp_std::collections::btree_set::BTreeSet;

    #[pallet::pallet]
    #[pallet::generate_store(pub (super) trait Store)]
    pub struct Pallet<T>(_);

    #[pallet::config]
    #[pallet::disable_frame_system_supertrait_check]
    //实现nft和timestamp的config
    pub trait Config: orml_nft::Config<TokenData=Kitty, ClassData=()> + pallet_timestamp::Config + SendTransactionTypes<Call<Self>> {
        type Randomness: Randomness<Self::Hash>;
        type Currency: Currency<Self::AccountId>;
        type WeightInfo: WeightInfo;

        #[pallet::constant]
        //难度系数
        type DefaultDifficulty: Get<u32>;

        type Event: From<Event<Self>> + IsType<<Self as frame_system::Config>::Event>;
    }


    #[pallet::storage]
    #[pallet::getter(fn kitty_info)]
    /// 根据TokenId获取kitty详情
    pub type KittyInfos<T: Config> = StorageMap<_,
        Blake2_128Concat,
        KittyIndexOf<T>,
        KittyDto<KittyIndexOf<T>, BalanceOf<T>, MomentOf<T>>,
        ValueQuery>;

    #[pallet::storage]
    #[pallet::getter(fn owned_kitties)]
    /// 账户所拥有的Kitty集合list<TokenId>
    pub type OwnedKitties<T: Config> = StorageMap<_,
        Blake2_128Concat,
        T::AccountId,
        BTreeSet<KittyIndexOf<T>>,
        ValueQuery>;

    #[pallet::storage]
    #[pallet::getter(fn kitty_sale_list)]
    /// 在售 kitty 集合 list<TokenId>.
    pub type KittySaleList<T: Config> = StorageValue<_, BTreeSet<KittyIndexOf<T>>, ValueQuery>;

    #[pallet::storage]
    #[pallet::getter(fn class_id)]
    /// The class id for orml_nft
    pub type ClassId<T: Config> = StorageValue<_, T::ClassId, ValueQuery>;

    #[pallet::storage]
    #[pallet::getter(fn auto_breed_nonce)]
    /// Nonce for auto breed to prevent replay attack
    pub type AutoBreedNonce<T: Config> = StorageValue<_, u32, ValueQuery>;

    #[pallet::storage]
    #[pallet::getter(fn kitty_difficulty_multiplier)]
    /// Difficulty multiplier which goes up the more kitty the account owns
    pub type KittyDifficultyMultiplier<T: Config> = StorageMap<_, Blake2_128Concat, KittyIndexOf<T>, u32, ValueQuery>;


    #[pallet::event]
    #[pallet::generate_deposit(pub (super) fn deposit_event)]
    #[pallet::metadata(T::AccountId = "AccountId")]
    pub enum Event<T: Config> {
        /// 一只新的kitty被创建. \[owner, kitty_id, kitty\]
        KittyCreated(T::AccountId, KittyIndexOf<T>, Kitty),
        /// 一只新的 kitten 出生. \[owner, kitty_id, kitty\]
        KittyBred(T::AccountId, KittyIndexOf<T>, Kitty),
        /// 一只kitty被赠送. \[from, to, kitty_id\]
        KittyTransferred(T::AccountId, T::AccountId, KittyIndexOf<T>),
        /// kitty价格更新. \[owner, kitty_id, price\]
        KittyPriceUpdated(T::AccountId, KittyIndexOf<T>, BalanceOf<T>),
        /// 一只kitty售出. \[old_owner, new_owner, kitty_id, price\]
        KittySold(T::AccountId, T::AccountId, KittyIndexOf<T>, BalanceOf<T>),
    }

    #[pallet::error]
    pub enum Error<T> {
        /// kittyId错误
        InvalidKittyId,
        /// 性别相同
        SameGender,
        /// 不是Kitty的主任
        NotOwner,
        /// 不可出售状态
        NotForSale,
        /// 出售中，不可赠送
        NotForTransfer,
        /// 价格太低
        PriceTooLow,
        /// 无需购买自己的Kitty
        BuyFromSelf,
        /// 难度系数溢出
        KittyDifficultyOverflow,
        /// 拥有数量溢出
        KittyOverflow,
        /// KittyId已经存在
        OwnedKittyId,
    }

    #[pallet::hooks]
    impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
        fn offchain_worker(_now: T::BlockNumber) {
            let _ = Self::run_offchain_worker();
        }
    }

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// 创建一只  kitty NFT
        #[pallet::weight(< T as pallet::Config >::WeightInfo::create())]
        pub(super) fn create(origin: OriginFor<T>) -> DispatchResultWithPostInfo {
            let sender = ensure_signed(origin)?;
            let dna = Self::random_value(&sender);
            // Create and store kitty
            let kitty = Kitty(dna);
            let kitty_id: KittyIndexOf<T> = NftModule::<T>::mint(&sender, Self::class_id(), Vec::new(), kitty.clone())?;
            // 获取当前区块高度 u32
            // let current_block = <frame_system::Module<T>>::block_number();
            // 当前时间戳 u64
            let _now = <timestamp::Module<T>>::get();

            let mut kitty_dto: KittyDto<KittyIndexOf<T>, BalanceOf<T>, MomentOf<T>> = KittyDto::default();
            kitty_dto.id = kitty_id;
            kitty_dto.dna = dna;
            kitty_dto.sex = kitty.gender();
            kitty_dto.birthday = _now;

            KittyInfos::<T>::insert(&kitty_id, kitty_dto);

            let mut owned_kitty = BTreeSet::new();
            if let Ok(list) = OwnedKitties::<T>::try_get(&sender) {
                owned_kitty = list;
            }
            owned_kitty.insert(kitty_id);
            OwnedKitties::<T>::insert(&sender, owned_kitty);

            // Emit event
            Self::deposit_event(crate::Event::KittyCreated(sender, kitty_id, kitty));

            Ok(().into())
        }


        /// 培育 kitties
        #[pallet::weight(< T as pallet::Config >::WeightInfo::breed())]
        pub(super) fn breed(origin: OriginFor<T>, kitty_id_1: KittyIndexOf<T>, kitty_id_2: KittyIndexOf<T>) -> DispatchResultWithPostInfo {
            let sender = ensure_signed(origin)?;

            let kitty1 = Self::kitties(&sender, kitty_id_1).ok_or(Error::<T>::InvalidKittyId)?;
            let kitty2 = Self::kitties(&sender, kitty_id_2).ok_or(Error::<T>::InvalidKittyId)?;

            Self::do_breed(sender, kitty1, kitty2)?;

            Ok(().into())
        }

        /// 赠送 kitty其他人
        #[pallet::weight(< T as pallet::Config >::WeightInfo::transfer())]
        pub fn transfer(origin: OriginFor<T>, to: T::AccountId, kitty_id: KittyIndexOf<T>) -> DispatchResultWithPostInfo {
            let sender = ensure_signed(origin)?;

            let info = KittyInfos::<T>::try_get(kitty_id).unwrap();

            //出售中的Kitty不可赠送
            ensure!(info.sale_status, Error::<T>::NotForTransfer);

            NftModule::<T>::transfer(&sender, &to, (Self::class_id(), kitty_id))?;

            //TODO 1.在售的kitty，不能赠送  2.old-owned移除，new-owned 新增
            if sender != to {
                // KittyInfos::<T>::remove(kitty_id);

                Self::deposit_event(Event::KittyTransferred(sender, to, kitty_id));
            }

            Ok(().into())
        }

        /// 设置Kitty的价格，更改出售状态，加入Kitty在售集合
        #[pallet::weight(< T as pallet::Config >::WeightInfo::set_price())]
        pub fn set_price(origin: OriginFor<T>, kitty_id: KittyIndexOf<T>, new_price: BalanceOf<T>) -> DispatchResultWithPostInfo {
            let sender = ensure_signed(origin)?;

            ensure!(orml_nft::TokensByOwner::<T>::contains_key(&sender, (Self::class_id(), kitty_id)), Error::<T>::NotOwner);

            // 1.加入在售列表 2.修改价格和Kitty_status
            let sale_status = KittyInfos::<T>::try_mutate_exists(kitty_id, |kitty_dto| -> Result<bool, DispatchError> {
                let info = kitty_dto.as_mut().ok_or(Error::<T>::InvalidKittyId)?;
                info.price = new_price;
                info.sale_status = !info.sale_status;
                Ok(info.sale_status)
            });

            let mut sale_list = BTreeSet::new();
            // 2.根据sale_status,操作在售Kitty集合
            if let Ok(list) = KittySaleList::<T>::try_get() {
                sale_list = list;
            }

            if let Ok(status) = sale_status {
                if status {
                    sale_list.insert(kitty_id);
                } else {
                    sale_list.remove(&kitty_id);
                }
            }

            KittySaleList::<T>::put(sale_list);

            Self::deposit_event(Event::KittyPriceUpdated(sender, kitty_id, new_price));

            Ok(().into())
        }

        /// 购买Kitty 支付Kitty所需的Token
        #[pallet::weight(< T as pallet::Config >::WeightInfo::buy())]
        pub fn buy(origin: OriginFor<T>, owner: T::AccountId, kitty_id: KittyIndexOf<T>, max_price: BalanceOf<T>) -> DispatchResultWithPostInfo {
            let sender = ensure_signed(origin)?;

            ensure!(sender != owner, Error::<T>::BuyFromSelf);

            KittyInfos::<T>::try_mutate_exists(kitty_id, |kitty_dto| -> DispatchResult {
                let kitty_dto = kitty_dto.take().ok_or(Error::<T>::NotForSale)?;

                ensure!(max_price >= kitty_dto.price, Error::<T>::PriceTooLow);

                with_transaction_result(|| {
                    NftModule::<T>::transfer(&owner, &sender, (Self::class_id(), kitty_id))?;
                    T::Currency::transfer(&sender, &owner, kitty_dto.price, ExistenceRequirement::KeepAlive)?;

                    Self::deposit_event(Event::KittySold(owner, sender, kitty_id, kitty_dto.price));

                    Ok(())
                })
            })?;

            Ok(().into())
        }

        #[pallet::weight(1000)]
        /// 拥有两只异性Kitty，自动培育下一代
        pub fn auto_breed(origin: OriginFor<T>, kitty_id_1: KittyIndexOf<T>, kitty_id_2: KittyIndexOf<T>, _nonce: u32, _solution: u128) -> DispatchResultWithPostInfo {
            ensure_none(origin)?;

            ensure!(Self::kitty_difficulty_multiplier(kitty_id_1) < u32::MAX, Error::<T>::KittyDifficultyOverflow);
            ensure!(Self::kitty_difficulty_multiplier(kitty_id_2) < u32::MAX, Error::<T>::KittyDifficultyOverflow);

            let kitty1 = NftModule::<T>::tokens(Self::class_id(), kitty_id_1).ok_or(Error::<T>::InvalidKittyId)?;
            let kitty2 = NftModule::<T>::tokens(Self::class_id(), kitty_id_2).ok_or(Error::<T>::InvalidKittyId)?;

            Self::do_breed(kitty1.owner, kitty1.data, kitty2.data)?;

            // Now that the parents have bred, increase difficulty multipler
            // default value is 0, so it starts at 0
            KittyDifficultyMultiplier::<T>::mutate(kitty_id_1, |multiplier| *multiplier = multiplier.saturating_add(1));
            KittyDifficultyMultiplier::<T>::mutate(kitty_id_2, |multiplier| *multiplier = multiplier.saturating_add(1));

            Ok(().into())
        }
    }


    #[cfg(feature = "std")]
    impl GenesisConfig {
        /// Direct implementation of `GenesisBuild::build_storage`.
        ///
        /// Kept in order not to break dependency.
        pub fn build_storage<T: Config>(&self) -> Result<sp_runtime::Storage, String> {
            <Self as GenesisBuild<T>>::build_storage(self)
        }

        /// Direct implementation of `GenesisBuild::assimilate_storage`.
        ///
        /// Kept in order not to break dependency.
        pub fn assimilate_storage<T: Config>(
            &self,
            storage: &mut sp_runtime::Storage,
        ) -> Result<(), String> {
            <Self as GenesisBuild<T>>::assimilate_storage(self, storage)
        }
    }

    #[pallet::genesis_config]
    #[derive(Default)]
    pub struct GenesisConfig {}

    #[pallet::genesis_build]
    impl<T: Config> GenesisBuild<T> for GenesisConfig {
        fn build(&self) {
            let class_id = NftModule::<T>::create_class(&Default::default(), Vec::new(), ()).expect("Cannot fail or invalid chain spec");
            ClassId::<T>::put(class_id);
        }
    }
}

//××私有方法××
/// 组合dna，dna可继承父母，也可能是变异
fn combine_dna(dna1: u8, dna2: u8, selector: u8) -> u8 {
    (!selector & dna1) | (selector & dna2)
}

impl<T: Config> Pallet<T> {
    fn kitties(owner: &T::AccountId, kitty_id: KittyIndexOf<T>) -> Option<Kitty> {
        NftModule::<T>::tokens(Self::class_id(), kitty_id).and_then(|x| {
            if x.owner == *owner {
                Some(x.data)
            } else {
                None
            }
        })
    }

    /// 生成随机hash
    fn random_value(sender: &T::AccountId) -> [u8; 16] {
        let payload = (
            T::Randomness::random_seed(),
            &sender,
            <frame_system::Module<T>>::extrinsic_index(),
        );
        payload.using_encoded(blake2_128)
    }

    /// 培育下一代Kitty
    fn do_breed(
        owner: T::AccountId,
        kitty1: Kitty,
        kitty2: Kitty,
    ) -> DispatchResult {
        ensure!(kitty1.gender() != kitty2.gender(), Error::<T>::SameGender);

        let kitty1_dna = kitty1.0;
        let kitty2_dna = kitty2.0;

        let selector = Self::random_value(&owner);
        let mut new_dna = [0u8; 16];

        // Combine parents and selector to create new kitty
        for i in 0..kitty1_dna.len() {
            new_dna[i] = combine_dna(kitty1_dna[i], kitty2_dna[i], selector[i]);
        }

        let new_kitty = Kitty(new_dna);
        let kitty_id = NftModule::<T>::mint(&owner, Self::class_id(), Vec::new(), new_kitty.clone())?;

        Self::deposit_event(Event::KittyBred(owner, kitty_id, new_kitty));

        Ok(())
    }

    fn validate_solution(kitty_id_1: KittyIndexOf<T>, kitty_id_2: KittyIndexOf<T>, nonce: u32, solution: u128) -> bool {
        let payload = (kitty_id_1, kitty_id_2, nonce, solution);
        let hash = payload.using_encoded(blake2_128);
        let hash_value = u128::from_le_bytes(hash);
        let multiplier = Self::kitty_difficulty_multiplier(kitty_id_1) + Self::kitty_difficulty_multiplier(kitty_id_2) + 1;
        let difficulty = multiplier * T::DefaultDifficulty::get();

        hash_value < (u128::max_value() / difficulty as u128)
    }

    fn run_offchain_worker() -> Result<(), ()> {
        let mut lock = StorageLock::<'_, BlockAndTime<frame_system::Module<T>>>::with_block_deadline(&b"kitties/lock"[..], 1);
        let _guard = lock.try_lock().map_err(|_| ())?;

        let random_seed = sp_io::offchain::random_seed();
        let mut rng = RandomNumberGenerator::<BlakeTwo256>::new(random_seed.into());

        // this only support if kitty_count <= u32::max_value()
        let kitty_count = TryInto::<u32>::try_into(orml_nft::Module::<T>::next_token_id(Self::class_id())).map_err(|_| ())?;

        const MAX_ITERATIONS: u128 = 500;

        let nonce = Self::auto_breed_nonce();

        let mut remaining_iterations = MAX_ITERATIONS;

        let (kitty_1, kitty_2) = loop {
            let kitty_id_1: KittyIndexOf<T> = rng.pick_u32(kitty_count).into();
            let kitty_id_2: KittyIndexOf<T> = rng.pick_u32(kitty_count).into();

            let kitty_1 = NftModule::<T>::tokens(Self::class_id(), kitty_id_1).ok_or(())?;
            let kitty_2 = NftModule::<T>::tokens(Self::class_id(), kitty_id_2).ok_or(())?;

            if kitty_1.data.gender() != kitty_2.data.gender() {
                break (kitty_id_1, kitty_id_2);
            }

            remaining_iterations -= 1;

            if remaining_iterations == 0 {
                return Err(());
            }
        };

        let solution_prefix = rng.pick_u32(u32::max_value() - 1) as u128;

        for i in 0..remaining_iterations {
            let solution = (solution_prefix << 32) + i;
            if Self::validate_solution(kitty_1, kitty_2, nonce, solution) {
                let _ = SubmitTransaction::<T, Call<T>>::submit_unsigned_transaction(Call::<T>::auto_breed(kitty_1, kitty_2, nonce, solution).into());
                break;
            }
        }

        Ok(())
    }
}

impl<T: Config> frame_support::unsigned::ValidateUnsigned for Module<T> {
    type Call = Call<T>;

    fn validate_unsigned(_source: TransactionSource, call: &Self::Call) -> TransactionValidity {
        match *call {
            Call::auto_breed(kitty_id_1, kitty_id_2, nonce, solution) => {
                if Self::validate_solution(kitty_id_1, kitty_id_2, nonce, solution) {
                    if nonce != Self::auto_breed_nonce() {
                        return InvalidTransaction::BadProof.into();
                    }

                    AutoBreedNonce::<T>::mutate(|nonce| *nonce = nonce.saturating_add(1));

                    ValidTransaction::with_tag_prefix("kitties")
                        .longevity(64_u64)
                        .propagate(true)
                        .build()
                } else {
                    InvalidTransaction::BadProof.into()
                }
            }
            _ => InvalidTransaction::Call.into(),
        }
    }
}

