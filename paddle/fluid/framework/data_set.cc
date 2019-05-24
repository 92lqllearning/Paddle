/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#include "paddle/fluid/framework/data_set.h"
#include <random>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/platform/timer.h"

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

namespace paddle {
namespace framework {

// constructor
template <typename T>
DatasetImpl<T>::DatasetImpl() {
  thread_num_ = 1;
  trainer_num_ = 1;
  file_idx_ = 0;
  input_channel_ = paddle::framework::make_channel<T>();
//  output_channel_ = paddle::framework::make_channel<T>();
}

// set filelist, file_idx_ will reset to zero.
template <typename T>
void DatasetImpl<T>::SetFileList(const std::vector<std::string>& filelist) {
  VLOG(3) << "filelist size: " << filelist.size();
  filelist_ = filelist;
  file_idx_ = 0;
}

// set expect thread num. actually it may change
template <typename T>
void DatasetImpl<T>::SetThreadNum(int thread_num) {
  VLOG(3) << "SetThreadNum thread_num=" << thread_num;
  thread_num_ = thread_num;
}

// if you run distributed, and want to do global shuffle,
// set this before global shuffle.
// be sure you call CreateReaders before SetTrainerNum
template <typename T>
void DatasetImpl<T>::SetTrainerNum(int trainer_num) {
  trainer_num_ = trainer_num;
  // should inform reader of trainer_num directly
  for (auto reader : readers_) {
    reader->SetTrainerNum(trainer_num);
  }
}

// if you run distributed, and want to do global shuffle,
// set this before global shuffle.
// be sure you call CreateReaders before SetFleetSendBatchSize
template <typename T>
void DatasetImpl<T>::SetFleetSendBatchSize(int64_t size) {
  fleet_send_batch_size_ = size;
  for (auto reader : readers_) {
    reader->SetFleetSendBatchSize(size);
  }
}

template <typename T>
void DatasetImpl<T>::SetHdfsConfig(const std::string& fs_name,
                                   const std::string& fs_ugi) {
  fs_name_ = fs_name;
  fs_ugi_ = fs_ugi;
  std::string cmd = std::string("hadoop fs");
  cmd += " -D fs.default.name=" + fs_name;
  cmd += " -D hadoop.job.ugi=" + fs_ugi;
  paddle::framework::hdfs_set_command(cmd);
}

template <typename T>
void DatasetImpl<T>::SetDataFeedDesc(const std::string& data_feed_desc_str) {
  google::protobuf::TextFormat::ParseFromString(data_feed_desc_str,
                                                &data_feed_desc_);
}

// readers_.size() may not be equal to thread_num_,
// it changes when filelist_.size() < thread_num_
template <typename T>
std::vector<std::shared_ptr<paddle::framework::DataFeed>>&
DatasetImpl<T>::GetReaders() {
  return readers_;
}

// if sent message between workers, should first call this function
template <typename T>
void DatasetImpl<T>::RegisterClientToClientMsgHandler() {
  auto fleet_ptr = FleetWrapper::GetInstance();
  VLOG(3) << "RegisterClientToClientMsgHandler";
  fleet_ptr->RegisterClientToClientMsgHandler(
      0, [this](int msg_type, int client_id, const std::string& msg) -> int {
        return this->ReceiveFromClient(msg_type, client_id, msg);
      });
  VLOG(3) << "RegisterClientToClientMsgHandler done";
}

// load data into memory, Dataset hold this memory,
// which will later be fed into readers' channel
template <typename T>
void DatasetImpl<T>::LoadIntoMemory() {
  int pid = (int)getpid();
  VLOG(0) << "DatasetImpl<T>::LoadIntoMemory() begin";
  VLOG(0) << shell_get_command_output(std::string("cat /proc/") + std::to_string(pid) + "/status | grep VmRSS");

  platform::Timer timeline;
  timeline.Start();
  if (readers_.size() == 0) {
    CreateReaders();
  }

  { 
     paddle::framework::Channel<T> tmp_input_channel = paddle::framework::make_channel<T>();
      for (int i = 0; i < thread_num_; ++i) {
        readers_[i]->SetInputChannel(tmp_input_channel);
      }

      std::vector<std::thread> load_threads;
      for (int64_t i = 0; i < thread_num_; ++i) {
        load_threads.push_back(std::thread(
            &paddle::framework::DataFeed::LoadIntoMemory, readers_[i].get()));
      }
      for (std::thread& t : load_threads) {
        t.join();
      }
      tmp_input_channel->close();
      timeline.Pause();

      VLOG(0) << "DatasetImpl<T>::LoadIntoMemory() end"
              << ", memory data size=" << tmp_input_channel->size()
              << ", cost time=" << timeline.ElapsedSec() << " seconds";

      VLOG(0) << shell_get_command_output(std::string("cat /proc/") + std::to_string(pid) + "/status | grep VmRSS");

      std::vector<std::thread>().swap(load_threads);
      std::vector<std::shared_ptr<paddle::framework::DataFeed>>().swap(readers_);

      VLOG(0) << "tmp_input_channel size=" << tmp_input_channel->size();

      VLOG(0) << "call clear";
      tmp_input_channel->my_clear();
      VLOG(0) << "call clear  end";

      VLOG(0) << "tmp_input_channel size=" << tmp_input_channel->size();

      tmp_input_channel = nullptr;

      VLOG(0) << "after tmp_input_channel clear " << shell_get_command_output(std::string("cat /proc/") + std::to_string(pid) + "/status | grep VmRSS");
  }
  VLOG(0) << "out of  tmp_input_channel scope " << shell_get_command_output(std::string("cat /proc/") + std::to_string(pid) + "/status | grep VmRSS");

}

// release memory data
template <typename T>
void DatasetImpl<T>::ReleaseMemory() {
  VLOG(3) << "DatasetImpl<T>::ReleaseMemory() begin";
  std::vector<T>().swap(memory_data_);
  for (int i = 0; i < readers_.size(); ++i) {
    readers_[i]->ReleaseChannelData();
  }
  VLOG(3) << "DatasetImpl<T>::ReleaseMemory() end";
}

// do local shuffle
template <typename T>
void DatasetImpl<T>::LocalShuffle() {
  VLOG(3) << "DatasetImpl<T>::LocalShuffle() begin";
  platform::Timer timeline;
  timeline.Start();
  if (readers_.size() == 0) {
    CreateReaders();
  }
  // if it is not InMemory, memory_data_ is empty
  std::random_shuffle(memory_data_.begin(), memory_data_.end());

  std::vector<std::thread> local_shuffle_threads;
  for (int64_t i = 0; i < thread_num_; ++i) {
    local_shuffle_threads.push_back(std::thread(
        &paddle::framework::DataFeed::LocalShuffle, readers_[i].get()));
  }
  for (std::thread& t : local_shuffle_threads) {
    t.join();
  }
  std::vector<T>().swap(memory_data_);
  timeline.Pause();
  VLOG(3) << "DatasetImpl<T>::LocalShuffle() end, cost time="
          << timeline.ElapsedSec() << " seconds";
}

template <typename T>
void DatasetImpl<T>::GlobalShuffle() {
  int pid = (int)getpid();
  VLOG(0) << "DatasetImpl<T>::GlobalShuffle() begin";
  VLOG(0) << shell_get_command_output(std::string("cat /proc/") + std::to_string(pid) + "/status | grep VmRSS");
  platform::Timer timeline;
  timeline.Start();
//  if (readers_.size() == 0) {
//    CreateReaders();
//  }

//  VLOG(0) << "open";
//  input_channel_->open();
 
/*  VLOG(0) << "channel size " << input_channel_->size();
  input_channel_->close();
   VLOG(0) <<"read";
  input_channel_->read_all(memory_data_);
  VLOG(0) << shell_get_command_output(std::string("cat /proc/") + std::to_string(pid) + "/status | grep VmRSS");
  VLOG(0) << "channel size " << input_channel_->size();
  auto fleet_ptr = FleetWrapper::GetInstance();
  VLOG(0) <<"local shuffle";
  // local shuffle all data before global shuffle
  std::shuffle(memory_data_.begin(), memory_data_.end(),
               fleet_ptr->LocalRandomEngine());
  VLOG(0) << shell_get_command_output(std::string("cat /proc/") + std::to_string(pid) + "/status | grep VmRSS");
  VLOG(0) << "write";
  input_channel_->open();
  input_channel_->write(std::move(memory_data_));*/
  VLOG(0) << "channel size " << input_channel_->size();
  VLOG(0) << shell_get_command_output(std::string("cat /proc/") + std::to_string(pid) + "/status | grep VmRSS");
//  input_channel_->close();
  VLOG(0) << "set_block_size";
  input_channel_->set_block_size(fleet_send_batch_size_);
   VLOG(0) << "close";
  input_channel_->close();
  VLOG(0) << "channel size " << input_channel_->size();
  VLOG(0) << shell_get_command_output(std::string("cat /proc/") + std::to_string(pid) + "/status | grep VmRSS");


  output_channel_vec_.resize(thread_num_);
  for (int i = 0; i < thread_num_; ++i) {
    output_channel_vec_[i] = paddle::framework::make_channel<T>();   
  }
  

  VLOG(3) << "start global shuffle threads";
  std::vector<std::thread> global_shuffle_threads;
  for (int i = 0; i < thread_num_; ++i) {
    global_shuffle_threads.push_back(std::thread(
//        &paddle::framework::DataFeed::GlobalShuffle, readers_[i].get()));
    [this]() {
   auto fleet_ptr = FleetWrapper::GetInstance();
  std::vector<paddle::ps::BinaryArchive> ars(this->trainer_num_);
  std::vector<T> data;
  while (this->input_channel_->read(data)) {
      VLOG(0) << "input_channel_->read(data)";
    for (auto& t : data) {
      auto client_id = fleet_ptr->LocalRandomEngine()() % this->trainer_num_;
      ars[client_id] << t;
    }

    std::vector<std::future<int32_t>> total_status;
    std::vector<int> send_index(this->trainer_num_);
    for (int i = 0; i < this->trainer_num_; ++i) {
      send_index[i] = i;
    }
    std::shuffle(send_index.begin(), send_index.end(), fleet_ptr->LocalRandomEngine());

    for (auto index = 0u; index < this->trainer_num_; ++index) {
      int i = send_index[index];
      if (ars[i].length() == 0) {
        continue;
      }
      std::string msg(ars[i].buffer(), ars[i].length());
      auto ret = fleet_ptr->SendClientToClientMsg(0, i, msg);
      total_status.push_back(std::move(ret));
    }

    for (auto& t : total_status) {
      t.wait();
    }

    ars.clear();
    ars = std::vector<paddle::ps::BinaryArchive>(this->trainer_num_);
    data = std::vector<T>();
    sleep(2);//this->fleet_send_sleep_seconds_);

  } 
    }));    
  }
  for (std::thread& t : global_shuffle_threads) {
    t.join();
  }
  VLOG(0) << "end global shuffle threads "
  << shell_get_command_output(std::string("cat /proc/") + std::to_string(pid) + "/status | grep VmRSS");

  std::vector<std::thread>().swap(global_shuffle_threads);
  VLOG(0) << "input channel size " << input_channel_->size();
  VLOG(0) << " mem size " << memory_data_.size();
  input_channel_->close();
//  input_channel_->read_all(memory_data_);
//  VLOG(0) << "input channel size " << input_channel_->size();
//  VLOG(0) << " mem size " << memory_data_.size();
//  std::vector<T>().swap(memory_data_);
//  VLOG(0) << "after swap  mem size " << memory_data_.size();
  timeline.Pause();
  input_channel_ = paddle::framework::make_channel<T>();
   VLOG(0) << "after reset input channel size " << input_channel_->size();
  VLOG(0) << "DatasetImpl<T>::GlobalShuffle() end, cost time="
          << timeline.ElapsedSec() << " seconds";
  VLOG(0) << shell_get_command_output(std::string("cat /proc/") + std::to_string(pid) + "/status | grep VmRSS");
}

template <typename T>
void DatasetImpl<T>::CreateReaders() {
  VLOG(3) << "Calling CreateReaders()";
  CHECK(thread_num_ > 0) << "thread_num should > 0";
  int file_cnt = filelist_.size();
  int memory_data_size = memory_data_.size();
  if (memory_data_size != 0 && thread_num_ > memory_data_size) {
    VLOG(3) << "Dataset thread num = " << thread_num_
            << ", memory data size = " << memory_data_size
            << ". Changing Dataset thread num = " << memory_data_size;
    thread_num_ = memory_data_size;
  } else if (file_cnt != 0 && thread_num_ > file_cnt) {
    VLOG(3) << "Dataset thread num = " << thread_num_
            << ", file num = " << file_cnt
            << ". Changing Dataset thread num = " << file_cnt;
    thread_num_ = file_cnt;
  }
  VLOG(3) << "thread_num in Readers: " << thread_num_;
  VLOG(3) << "readers size: " << readers_.size();
  VLOG(3) << "Filelist size in readers: " << filelist_.size();
  if (readers_.size() != 0) {
    return;
  }
  VLOG(3) << "data feed class name: " << data_feed_desc_.name();
  for (int i = 0; i < thread_num_; ++i) {
    readers_.push_back(DataFeedFactory::CreateDataFeed(data_feed_desc_.name()));
    readers_.back()->Init(data_feed_desc_);
    readers_.back()->SetMemoryData(&memory_data_);
    readers_.back()->SetMemoryDataMutex(&mutex_for_update_memory_data_);
    readers_.back()->SetThreadId(i);
    readers_.back()->SetThreadNum(thread_num_);
    readers_.back()->SetTrainerNum(trainer_num_);
    readers_.back()->SetFileListMutex(&mutex_for_pick_file_);
    readers_.back()->SetFileListIndex(&file_idx_);
    readers_.back()->SetFileList(filelist_);
//    readers_.back()->SetInputChannel(input_channel_);
  }
}

template <typename T>
void DatasetImpl<T>::DestroyReaders() {
  VLOG(3) << "Calling DestroyReaders()";
  // clear memory_data_ before fill it
  // because if LoadIntoMemory but no Shuffle,
  // memory_data_ has empty data which has been std::move to channel
  if (memory_data_.size() != 0) {
    std::vector<T>().swap(memory_data_);
  }
  std::vector<std::thread> fill_threads;
  for (int i = 0; i < thread_num_; ++i) {
    fill_threads.push_back(
        std::thread(&paddle::framework::DataFeed::FillChannelToMemoryData,
                    readers_[i].get()));
  }
  for (std::thread& t : fill_threads) {
    t.join();
  }
  std::vector<std::shared_ptr<paddle::framework::DataFeed>>().swap(readers_);
  VLOG(3) << "readers size: " << readers_.size();
  // if memory_data_ is empty, which means it's not InMemory mode,
  // so the next epoch should read all data again
  if (memory_data_.size() == 0) {
    file_idx_ = 0;
  }
}

template <typename T>
int64_t DatasetImpl<T>::GetMemoryDataSize() {
  return memory_data_.size();
}

template <typename T>
int64_t DatasetImpl<T>::GetShuffleDataSize() {
  int64_t sum = 0;
  for (int i = 0; i < readers_.size(); ++i) {
    sum += readers_[i]->GetChannelDataSize();
  }
  return sum;
}

template <typename T>
int DatasetImpl<T>::ReceiveFromClient(int msg_type, int client_id,
                                      const std::string& msg) {
#ifdef _LINUX
  VLOG(0) << "ReceiveFromClient msg_type=" << msg_type
          << ", client_id=" << client_id << ", msg length=" << msg.length();
  auto fleet_ptr = FleetWrapper::GetInstance();
  int64_t index = fleet_ptr->LocalRandomEngine()() % thread_num_;
  VLOG(0) << "ramdom index=" << index;

  if (msg.length() == 0) {
    return 0;
  }

  paddle::ps::BinaryArchive ar;
    ar.set_read_buffer(const_cast<char*>(msg.c_str()), msg.length(), nullptr);
    if (ar.cursor() == ar.finish()) {
        return 0;
    }
    std::vector<T> data;
    while (ar.cursor() < ar.finish()) {
        data.push_back(ar.get<T>());
    }
    CHECK(ar.cursor() == ar.finish());
    output_channel_vec_[index]->write(std::move(data));
//    output_channel_vec_[0]->write(std::move(data));

//  readers_[index]->PutInsToChannel(msg);
#endif
  return 0;
}

// explicit instantiation
template class DatasetImpl<std::vector<MultiSlotType>>;

}  // end namespace framework
}  // end namespace paddle
